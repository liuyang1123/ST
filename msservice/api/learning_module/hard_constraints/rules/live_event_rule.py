from .rule import Rule


class LiveEventRule(Rule):
    """
    Para los eventos que son presenciales.

    Check:
        - Timezone
        - Travel
    """

    def valid(self, event):
        """
        Retorna 1 si la condicion de la regla se cumple.
        Retorna 0 en otro caso.
        """
        if event.is_live():
            for attendee in event.participants:
                # Cities
                if not attendee.get_location(
                        event.start_time).compare(event.location):
                    return 0
        # TODO Revisar posibles problemas con diferentes timezones
        return 1

    def possible_solution(self, event):
        return None

    def has_possible_solution(self, event):
        return False
