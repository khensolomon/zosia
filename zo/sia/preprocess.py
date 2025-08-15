"""
Zolai-NMT Data Preprocessing Module
version: 2025.08.10.0821

This module contains all functions related to loading, generating, and preparing
data for the NMT model. It handles TSV file indexing, template-based data
generation with grammatical modifiers, and tokenizer data preparation.
"""
import os
import io
import random
import re
import itertools
import yaml
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# --- 1. Data Loading & Generation ---

def deep_merge_dicts(d1, d2):
    """
    Recursively merges dictionary d2 into d1. List values are extended,
    and dictionary values are recursively merged.
    """
    for k, v in d2.items():
        if k in d1:
            if isinstance(d1[k], dict) and isinstance(v, dict):
                deep_merge_dicts(d1[k], v)
            elif isinstance(d1[k], list) and isinstance(v, list):
                d1[k].extend(v)
            else:
                d1[k] = v # d2's value overrides d1's
        else:
            d1[k] = v
    return d1

def get_indexed_pairs(cfg, source_lang, target_lang, data_type='train'):
    """Loads parallel data from a list of TSV files defined in datasets.yaml."""
    index_file = cfg.data.paths.datasets_index_file
    if not os.path.exists(index_file):
        print(f"Warning: Datasets index file not found at {index_file}")
        return []

    with open(index_file, 'r', encoding='utf-8') as f:
        index = yaml.safe_load(f)

    direction_key = f"{source_lang}-{target_lang}"
    direction_config = index.get(direction_key, {})

    # Determine the correct directory for train/test files
    dir_key = f"{data_type}_dir"
    data_dir = index.get(dir_key, os.path.join(cfg.data.paths.corpus_dir, data_type))

    # Combine specific and shared basenames, ensuring uniqueness
    specific_basenames = direction_config.get(data_type, [])
    shared_basenames = index.get("shared", {}).get(data_type, [])
    basenames = list(dict.fromkeys(specific_basenames + shared_basenames))

    all_pairs = []
    print(f"Loading TSV parallel data for '{direction_key}' ({data_type} set)...")
    for item in basenames:
        file_path = os.path.join(data_dir, f"{item}.tsv")

        if not os.path.exists(file_path):
            print(f"  - Warning: File '{file_path}' not found. Skipping.")
            continue

        with io.open(file_path, encoding='utf-8') as f:
            try:
                header = f.readline().strip().split('\t')
                src_idx = header.index(source_lang)
                tgt_idx = header.index(target_lang)
                print(f"  - Loading from '{os.path.basename(file_path)}'...")
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) > max(src_idx, tgt_idx):
                        all_pairs.append((parts[src_idx], parts[tgt_idx]))
            except (ValueError, IndexError):
                print(f"  - Skipping '{os.path.basename(file_path)}' (missing required language columns or malformed line).")
                continue
    return all_pairs

def load_templated_data(cfg, source_lang, target_lang, args):
    """Generates sentence pairs by crawling and processing YAML template files."""
    template_dir = cfg.data.paths.template_dir
    if not os.path.exists(template_dir): return []

    # Safely get tag arguments, defaulting to None if not present
    include_tags_str = getattr(args, 'include_tags', None)
    exclude_tags_str = getattr(args, 'exclude_tags', None)
    
    include_tags = set(include_tags_str.split(',')) if include_tags_str else None
    exclude_tags = set(exclude_tags_str.split(',')) if exclude_tags_str else None

    all_pairs = []
    print(f"Loading templated data for '{source_lang}-{target_lang}'...")
    # Sort filenames to ensure deterministic order
    for filename in sorted(os.listdir(template_dir)):
        if not filename.endswith((".yaml", ".yml")): continue
        
        file_path = os.path.join(template_dir, filename)
        
        # Tag filtering logic
        if include_tags or exclude_tags:
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = yaml.safe_load(f)
                    file_tags = set(data.get('tags', []))
                    if include_tags and not file_tags.intersection(include_tags): 
                        continue
                    if exclude_tags and file_tags.intersection(exclude_tags):
                        continue
                except yaml.YAMLError: 
                    continue
        
        print(f"  - Generating from '{filename}'...")
        all_pairs.extend(generate_from_template_file(cfg, file_path, source_lang, target_lang))
    return all_pairs

def capitalize_sentence(sentence):
    """Capitalizes the first letter of a sentence, ignoring 'i'."""
    if not sentence: return ""
    parts = sentence.split(' ', 1)
    first_word = parts[0]
    if first_word.lower() != 'i':
        first_word = first_word.capitalize()
    return ' '.join([first_word] + parts[1:]) if len(parts) > 1 else first_word

def check_tag_condition(dependent_tags, required_condition):
    """
    Checks if a set of tags meets a required condition.
    Normalizes tags by treating hyphens and colons as the same separator.
    """
    normalized_tags = {tag.replace('-', ':') for tag in dependent_tags}

    if isinstance(required_condition, str):
        return required_condition.replace('-', ':') in normalized_tags
    if isinstance(required_condition, dict):
        if 'all_of' in required_condition:
            normalized_required = {req.replace('-', ':') for req in required_condition['all_of']}
            return normalized_required.issubset(normalized_tags)
        if 'any_of' in required_condition:
            normalized_required = {req.replace('-', ':') for req in required_condition['any_of']}
            return any(t in normalized_tags for t in normalized_required)
        if 'none_of' in required_condition:
            normalized_required = {req.replace('-', ':') for req in required_condition['none_of']}
            return not any(t in normalized_tags for t in normalized_required)
    return False

def resolve_template_string(template_str, context, template_data, lang):
    """Resolves a template string with advanced modifiers."""
    placeholder_regex = re.compile(r"<(\w+)(?:\[([\w,]+)\])?(?:\|(\w+)(?::(\w+))?)?>")

    def get_value_from_context(full_placeholder_key, language):
        """Safely gets a value from the context, using 'en' as a primary fallback."""
        if full_placeholder_key not in context:
            return ""
        data = context[full_placeholder_key]
        value = data.get(language, data.get('en', data.get('canonical_key', '')))
        return value

    def replacer(match):
        p_name, group, modifier, mod_arg = match.groups()
        full_placeholder_key = f"{p_name}[{group}]" if group else p_name
        
        # --- Handle Modifiers ---
        if modifier == 'form':
            if not mod_arg: return ""
            mod_arg_key = next((k for k in context if k.startswith(mod_arg)), None)
            if not mod_arg_key: return ""

            verb_canonical_key = context[mod_arg_key].get('canonical_key')
            verb_metadata = (template_data.get('metadata', {}).get(mod_arg) or {}).get(verb_canonical_key) or {}
            pronoun_group = verb_metadata.get('pronoun_group', 1)
            
            subject_forms = get_value_from_context(full_placeholder_key, lang)
            if isinstance(subject_forms, list):
                form_index = pronoun_group - 1
                if 0 <= form_index < len(subject_forms):
                    return subject_forms[form_index]
            return subject_forms

        elif modifier == 'plural':
            canonical_key = context[full_placeholder_key].get('canonical_key')
            plural_form = (template_data.get('metadata', {}).get(p_name, {}).get(canonical_key) or {}).get('plural', {}).get(lang)
            if plural_form: return plural_form
            
            base_word = get_value_from_context(full_placeholder_key, lang)
            if lang == 'en': return base_word + 's'
            return base_word
        
        elif modifier == 'article':
            if not mod_arg: return ""
            mod_arg_key = next((k for k in context if k.startswith(mod_arg)), None)
            if not mod_arg_key: return ""

            if lang == 'en':
                canonical_key = context[mod_arg_key].get('canonical_key')
                tags = set((template_data.get('metadata', {}).get(mod_arg, {}).get(canonical_key) or {}).get('tags', []))
                if 'starts_with_vowel' in tags: return "an"
                if 'starts_with_consonant' in tags: return "a"
                
                word = get_value_from_context(mod_arg_key, lang)
                return "an" if word and word.lower().startswith(('a', 'e', 'i', 'o', 'u')) else "a"
            return ""

        elif modifier == 'person':
            if not mod_arg: return get_value_from_context(full_placeholder_key, lang)
            mod_arg_key = next((k for k in context if k.startswith(mod_arg)), None)
            if not mod_arg_key: return get_value_from_context(full_placeholder_key, lang)
            
            raw_subject_tags = context[mod_arg_key].get('tags', [])
            subject_tags = {tag.replace('-', ':') for tag in raw_subject_tags}
            
            if 'person:third' in subject_tags and 'singular' in subject_tags:
                canonical_key = context[full_placeholder_key].get('canonical_key')
                third_person_form = (template_data.get('metadata', {}).get(p_name, {}).get(canonical_key) or {}).get('person:third_singular', {}).get(lang)
                if third_person_form: return third_person_form
                
                base_verb = get_value_from_context(full_placeholder_key, lang)
                if lang == 'en': return base_verb + 's'
                return base_verb
            
            return get_value_from_context(full_placeholder_key, lang)

        # --- Handle Standard Placeholders ---
        value = get_value_from_context(full_placeholder_key, lang)
        if isinstance(value, list): return value[0]
        return str(value)

    return placeholder_regex.sub(replacer, template_str)

def generate_from_template_file(cfg, file_path, source_lang, target_lang):
    try:
        with open(file_path, 'r', encoding='utf-8') as f: template_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Warning: Could not parse YAML file {file_path}. Error: {e}"); return []
    
    if 'import' in template_data:
        merged_data = {}
        for import_file in template_data['import']:
            import_path = os.path.join(cfg.data.paths.shared_template_dir, import_file)
            if os.path.exists(import_path):
                with open(import_path, 'r', encoding='utf-8') as f:
                    shared_data = yaml.safe_load(f)
                    deep_merge_dicts(merged_data, shared_data)
            else:
                print(f"Warning: Imported file not found: {import_path}")
        
        local_data_to_merge = {k: v for k, v in template_data.items() if k != 'import'}
        deep_merge_dicts(merged_data, local_data_to_merge)
        template_data = merged_data

    templates = template_data.get('templates', [])
    if not templates: return []

    generated_pairs = []
    for template_pair in templates:
        src_template = template_pair.get(source_lang)
        tgt_template = template_pair.get(target_lang)
        if not src_template or not tgt_template: continue
        
        full_placeholder_regex = re.compile(r"<(\w+(?:\[[\w,]+\])?)(?:\|.*?)?>")
        used_placeholders = sorted(list(set(full_placeholder_regex.findall(src_template + tgt_template))))
        used_placeholders = [p for p in used_placeholders if p.split('[')[0] != 'article']
        
        # Separate conditional from non-conditional placeholders
        conditional_placeholder_names = {p.split('[')[0] for p in used_placeholders if isinstance(template_data.get(p.split('[')[0]), dict) and 'default' in template_data.get(p.split('[')[0])}
        non_conditional_placeholders = [p for p in used_placeholders if p.split('[')[0] not in conditional_placeholder_names]

        if not non_conditional_placeholders and not conditional_placeholder_names:
            generated_pairs.append((capitalize_sentence(src_template), capitalize_sentence(tgt_template))); continue
        
        all_options = []
        valid_template = True
        for full_placeholder in non_conditional_placeholders:
            match = re.match(r"(\w+)(?:\[([\w,]+)\])?", full_placeholder)
            p_name, group_str = match.groups()
            
            group_filters = set(group_str.split(',')) if group_str else set()
            
            options_for_this_placeholder = []
            values = template_data.get(p_name)
            # if isinstance(values, dict):
            #     for canonical_key, data in values.items():
            #         if group_filters:
            #             item_groups = data.get('group', [])
            #             if isinstance(item_groups, str): item_groups = [item_groups]
            #             if not group_filters.issubset(set(item_groups)):
            #                 continue
                    
            #         option = data.copy()
            #         option['canonical_key'] = canonical_key
            #         options_for_this_placeholder.append(option)
            # else:
            #     valid_template = False; break

            if isinstance(values, dict):
                for canonical_key, data in values.items():
                    # Group filtering (supports multiple groups, all required)
                    if group_filters:
                        item_groups = data.get('group', [])
                        if isinstance(item_groups, str): item_groups = [item_groups]
                        if not group_filters.issubset(set(item_groups)): continue

                    option = data.copy()
                    option['canonical_key'] = canonical_key

                    # NEW: Merge in tags from metadata like the old function did
                    md_entry = ((template_data.get('metadata', {}) or {}).get(p_name, {}) or {}).get(canonical_key, {}) or {}
                    md_tags = md_entry.get('tags', [])
                    if md_tags:
                        option['tags'] = list(set(option.get('tags', [])) | set(md_tags))
                    options_for_this_placeholder.append(option)
            elif isinstance(values, list) and all(isinstance(v, str) for v in values):
                # NEW: Support simple lists like the old function
                for v in values:
                    option = {
                        source_lang: v,
                        target_lang: v,
                        'tags': [],
                        'canonical_key': v
                    }
                    options_for_this_placeholder.append(option)
            else:
                valid_template = False; break            
            
            if not options_for_this_placeholder: valid_template = False; break
            all_options.append(options_for_this_placeholder)

        if not valid_template: continue

        # Handle case where there are only conditional placeholders
        if not all_options and conditional_placeholder_names:
            combos = [()]
        else:
            combos = itertools.product(*all_options)

        for combo in combos:
            context = {non_conditional_placeholders[i]: combo[i] for i in range(len(combo))}
            
            for p_name in conditional_placeholder_names:
                p_def = template_data.get(p_name, {})
                src_val, tgt_val = p_def['default'][source_lang], p_def['default'][target_lang]
                
                for rule in p_def.get('rules', []):
                    conditions = rule.get('conditions', {})
                    if not conditions: continue
                    
                    all_conditions_met = True
                    for dependent_p_name, required_cond in conditions.items():
                        dep_full_key = next((k for k in context if k.startswith(dependent_p_name)), None)
                        if not dep_full_key:
                            all_conditions_met = False; break
                        
                        dependent_tags = set(context[dep_full_key].get('tags', []))
                        if not check_tag_condition(dependent_tags, required_cond):
                            all_conditions_met = False; break
                    
                    if all_conditions_met:
                        src_val = rule['translations'].get(source_lang, src_val)
                        tgt_val = rule['translations'].get(target_lang, tgt_val)
                        break
                
                context[p_name] = {source_lang: src_val, target_lang: tgt_val, 'canonical_key': p_name}

            final_src = resolve_template_string(src_template, context, template_data, source_lang)
            final_tgt = resolve_template_string(tgt_template, context, template_data, target_lang)

            generated_pairs.append((capitalize_sentence(final_src), capitalize_sentence(final_tgt)))
            
    return generated_pairs

# --- 2. Data Preparation & Datasets ---

def prepare_tokenizer_data(cfg, source_lang, target_lang, args):
    """Aggregates all specified data sources into a single text file for tokenizer training."""
    all_sentences = []
    if args.use_parallel:
        for src, tgt in [(source_lang, target_lang), (target_lang, source_lang)]:
            train_pairs = get_indexed_pairs(cfg, src, tgt, 'train')
            test_pairs = get_indexed_pairs(cfg, src, tgt, 'test')
            all_sentences.extend([p[0] for p in train_pairs] + [p[1] for p in train_pairs])
            all_sentences.extend([p[0] for p in test_pairs] + [p[1] for p in test_pairs])
    if args.use_template:
        for src, tgt in [(source_lang, target_lang), (target_lang, source_lang)]:
            template_pairs = load_templated_data(cfg, src, tgt, args)
            all_sentences.extend([p[0] for p in template_pairs] + [p[1] for p in template_pairs])

    # Sort the collected sentences to ensure a deterministic order for hashing
    all_sentences.sort()

    os.makedirs(cfg.data.paths.tmp_dir, exist_ok=True)
    lang_pair_str = f"{source_lang}-{target_lang}"
    # Use the nested config path: cfg.app.filenames
    output_path = os.path.join(cfg.data.paths.tmp_dir, cfg.app.filenames.tokenizer_input.format(lang_pair=lang_pair_str))
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_sentences))
    
    unique_words = len(set(" ".join(all_sentences).split()))
    return output_path, unique_words

def build_data_index(cfg, source_lang, target_lang, args):
    """Builds a combined index of sentence pairs from all specified sources."""
    index = []
    if args.use_parallel:
        index.extend(get_indexed_pairs(cfg, source_lang, target_lang, 'train'))
    if args.use_template:
        index.extend(load_templated_data(cfg, source_lang, target_lang, args))
    
    if not index:
        raise ValueError(f"No data found for the pair '{source_lang}-{target_lang}' from any source.")
    
    random.shuffle(index)
    return index

class StreamingTranslationDataset(Dataset):
    """A PyTorch Dataset that tokenizes sentence pairs on the fly."""
    def __init__(self, data_index, tokenizer, cfg):
        self.data_index = data_index
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        src, tgt = self.data_index[idx]
        # Use the nested config path: cfg.app.tokens
        src_ids = [self.cfg.app.tokens.sos] + self.tokenizer.encode(src) + [self.cfg.app.tokens.eos]
        tgt_ids = [self.cfg.app.tokens.sos] + self.tokenizer.encode(tgt) + [self.cfg.app.tokens.eos]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def collate_fn(batch, pad_token_id):
    """Pads sequences in a batch to the same length."""
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_token_id)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_token_id)
    return src_padded, tgt_padded
