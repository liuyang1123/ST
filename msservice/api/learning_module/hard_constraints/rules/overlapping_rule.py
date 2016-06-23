from .rule import Rule


class OverlappingRule(Rule):
    """
    Son condiciones que el evento y todos sus miembros deben cumplir.
    Ya que de no cumplirse ese individuo no podra asistir.
    """

    def valid(self, event):
        """
        Retorna 1 si la condicion de la regla se cumple.
        Retorna 0 en otro caso.
        """
        if event.exists_overlap():
            return 0
        return 1

    def possible_solution(self, event):
        """
        La solucion es buscar otro timeslot. Por definicion del problema no
        se tendria que llegar a encontrar un overlap.
        """
        return None

    def has_possible_solution(self, event):
        # Just move to another timeslot
        return False
