class Rule:
    """
    Son condiciones que el evento y todos sus miembros deben cumplir.
    Ya que de no cumplirse ese individuo no podra asistir.
    """

    def valid(self, event):
        """
        Retorna 1 si la condicion de la regla se cumple.
        Retorna 0 en otro caso.
        """
        return

    def possible_solution(self, event):
        """
        Para algunas reglas seria ideal ofrecer una accion para solucionar el
        conflicto que provoca que la condicion no se cumpla.
            - Por ejemplo si un invitado esta de viaje no tiene sentido seguir
            buscando timeslots antes de que el invitado retorne.
            Se lo excluye, o se cambia la fecha de la reunion.
        """
        return

    def has_possible_solution(self, event):
        return
