from .rule import Rule


class CheckInformationDisclosureRule(Rule):
    """
    Comprueba que todos los participantes puedan estar presentes en la reunion
    dependiendo de si el tema a tratar es confidencial.
    Por ejemplo: Competidores.

    TODO:
        - Usar mineria de datos
    """

    def valid(self, event):
        """
        Retorna 1 si la condicion de la regla se cumple.
        Retorna 0 en otro caso.
        """
        return 1

    def possible_solution(self, event):
        """
        Remove the user from the event.
        """
        return None

    def has_possible_solution(self, event):
        return False
