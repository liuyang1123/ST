from .check_information_disclosure import CheckInformationDisclosureRule
from .live_event_rule import LiveEventRule
from .moc_rule import ModeOfCommunicationRule
from .overlapping_rule import OverlappingRule
from .unique_per_day_rule import UniquePerDayRule


class RulesManager:

    def is_valid(self, event):
        r1 = CheckInformationDisclosureRule()
        r2 = LiveEventRule()
        r3 = ModeOfCommunicationRule()
        r4 = OverlappingRule()
        r5 = UniquePerDayRule()

        # TODO Use rule solutions

        return (r1.valid(event) and r2.valid(event) and
                r3.valid(event) and r4.valid(event) and
                r5.valid(event))
