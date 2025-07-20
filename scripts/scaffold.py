# org.py
# This file defines classes and/or logic related to organizations.
# Since the original file is empty, we will set up a basic structure
# that can be built upon, complete with inline comments for clarity.

class Organization:
    """
    Represents an organization with basic details such as name and type.
    """

    def __init__(self, name, org_type):
        # Store the name of the organization
        self.name = name

        # Store the type of organization (e.g., NGO, company, school)
        self.org_type = org_type

        # Initialize an empty list to keep track of members
        self.members = []

    def add_member(self, person):
        """
        Adds a person to the organization's member list.

        Args:
            person (str): Name or identifier of the person to add
        """
        # Prevent adding duplicates
        if person not in self.members:
            self.members.append(person)

    def remove_member(self, person):
        """
        Removes a person from the organization's member list if they exist.

        Args:
            person (str): Name or identifier of the person to remove
        """
        if person in self.members:
            self.members.remove(person)

    def get_member_count(self):
        """
        Returns the number of members in the organization.

        Returns:
            int: Number of current members
        """
        return len(self.members)

    def __str__(self):
        # Return a user-friendly string representation of the organization
        return f"{self.name} ({self.org_type}) with {len(self.members)} members"


# Example usage:
if __name__ == "__main__":
    # Create an instance of Organization
    org = Organization("Green Earth", "NGO")

    # Add some members
    org.add_member("Alice")
    org.add_member("Bob")

    # Print the organization details
    print(org)

    # Output: Green Earth (NGO) with 2 members
