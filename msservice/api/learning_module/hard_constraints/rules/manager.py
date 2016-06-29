from .check_information_disclosure import CheckInformationDisclosureRule
from .live_event_rule import LiveEventRule
from .moc_rule import ModeOfCommunicationRule
from .overlapping_rule import OverlappingRule
from .unique_per_day_rule import UniquePerDayRule


class RulesManager:

    def __init__(self):
        r1 = CheckInformationDisclosureRule()
        r2 = LiveEventRule()
        r3 = ModeOfCommunicationRule()
        r4 = OverlappingRule()
        r5 = UniquePerDayRule()
        self.rules = [r1, r2, r3, r4, r5]
        self.invalid_rules = []

    def is_valid(self, event):
        val = True
        self.invalid_rules = []
        for rule in self.rules:
            re = rule.valid(event)
            val = val and re
            if not re:
                self.invalid_rules.append(rule)

        return val

    def has_possible_solution(self, event):
        val = False
        for rule in self.invalid_rules:
            val = val or rule.has_possible_solution(event)

        return val

    def possible_solution(self, event):
        resulting_event = event
        for rule in self.invalid_rules:
            if not rule.valid(resulting_event) and rule.has_possible_solution(resulting_event):
                resulting_event = rule.possible_solution(resulting_event)

        return resulting_event
