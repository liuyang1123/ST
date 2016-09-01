class Location(object):
    """
    Name: e.g. "Room 2"
    Description: e.g. ""
    Capacity: e.g. 25
    Location: e.g. "74º...""

    Additional fields:
        - Equipment
        - Location can be the GPS coordinates
    """

    def __init__(self, name, description="", capacity=None, location=None):
        self.name = name
        self.description = description
        self.capacity = capacity
        self.location = location

    def is_free(self, start, end):
        pass

    def supports(self, number_of_people):
        if self.capacity is not None and self.capacity < number_of_people:
            return False
        return True

    def compare(self, location):
        # TODO Find a way to compare coordinates, or add a city field.
        return

    def distance(self, location):
        # TODO Figure out a way to do this calculation
        return
