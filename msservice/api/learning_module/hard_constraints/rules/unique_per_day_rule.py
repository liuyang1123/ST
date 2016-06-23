from .rule import Rule
import datetime


class UniquePerDayRule(Rule):
    """
    Esta condicion consiste en verificar que eventos que debieran ser unicos por
    dia se cumpla. Por ejemplo, dos almuerzos en un dia.
    """

    def valid(self, event):
        """
        Retorna 1 si la condicion de la regla se cumple.
        Retorna 0 en otro caso.

        Para seleccionar el metodo mas comun se retorna 0, de esta forma luego
        se invoca el metodo possible_solution.
        """
        if event.should_be_unique():
            for attendee in event.participants:
                if attendee.exists_event(event.event_type,
                                         event.get_day()):
                    return 0
        return 1

    def possible_solution(self, event):
        """
        Ofrece una fecha distinta, por ejemplo un dia despues.

        TODO:
            - Ofrecer el proximo dia apto. Por ejemplo si era sabado, un domingo
            no debe ser una buena opcion.
        """
        event.move_time(datetime.timedelta(days=1))
        return event

    def has_possible_solution(self, event):
        return True
