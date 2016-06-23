from .rule import Rule
from api.learning_module.hard_constraints.preferences.preference import MODE_OF_COMMUNICATION


class ModeOfCommunicationRule(Rule):
    """
    Esta condicion consiste en verificar que el medio utilizado sea el preferido
    por la mayoria de los invitados.
    """

    def valid(self, event):
        """
        Retorna 1 si la condicion de la regla se cumple.
        Retorna 0 en otro caso.

        Para seleccionar el metodo mas comun se retorna 0, de esta forma luego
        se invoca el metodo possible_solution.
        """
        # return 0
        return 1

    def possible_solution(self, event):
        """
        Para algunas reglas seria ideal ofrecer una accion para solucionar el
        conflicto que provoca que la condicion no se cumpla.
            - Por ejemplo si un invitado esta de viaje no tiene sentido seguir
            buscando timeslots antes de que el invitado retorne.
            Se lo excluye, o se cambia la fecha de la reunion.
        """
        invitees = event.participants
        moc_pref = dict()
        for i in invitees:
            p = i.get_preference(MODE_OF_COMMUNICATION)
            if p in moc_pref:
                moc_pref[p] += 1
            else:
                moc_pref[p] = 1

        # Set the mode of communication for the event
        event.set_mode_of_communication(max(moc_pref, key=moc_pref.get))
        return event

    def has_possible_solution(self, event):
        return True
