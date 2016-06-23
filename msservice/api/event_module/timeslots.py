from calendar_client import CalendarDBClient

# TODO Transfer this to calendar_client


class TimeSlot:
    __start = None
    __end = None

    def __init__(self, start=None, end=None):
        self.__start = start
        self.__end = end

    def get_start(self):
        return self.__start

    def set_start(self, start):
        self.__start = start

    def get_end(self):
        return self.__end

    def set_end(self, end):
        self.__end = end

    def to_json(self):
        """
        {"start": __start, "end": __end}
        """
        pass


class TimeSlotManager:
    """
    """

    def __init__(self, user_id, start_period, end_period, duration):
        self.__user_id = user_id
        self.__start_period = start_period  # Initial value of the interval
        self.__end_period = end_period  # Final value of the interval
        # self.__duration = duration
        self.__iterator = 0
        self.__db_client = CalendarDBClient()
        self.__time_slots = self.__filter()

        self.number_of_slots = duration / 5
        self.slot_delta = datetime.timedelta(
            seconds=(5 * 60))  # 5 Min = size of one slot
        self.day_delta = datetime.timedelta(days=1)
        self.are_more_slots = True

    def __filter(self):
        # TODO
        # Para filtrar se usan las hard constraints. Falta tener en cuenta L, M, M, J, V, S, D (cambiaria la forma de obtener los valores free_busy)
        # y se ordenan por las softconstraints
        # se tendria que pasar un parametro, de severidad (1=todo se tiene que
        # cumplir, 0.7=al no llegar a un acuerdo se trata de hacer cambios el
        # minimo del confidence score)
        self.__iterator = 0

        result = []
        free_space = []
        if self.__start_period + day_delta <= self.__end_period:
            free_space = self.__db_client.free_busy(
                self.__user_id, self.__start_period, self.__start_period + day_delta)
            self.__start_period = self.__start_period + day_delta
        else:
            free_space = self.__db_client.free_busy(
                self.__user_id, self.__start_period, self.__end_period)
            self.are_more_slots = False

        for space in free_space:
            start_date = parse(space['start'])
            end_date = parse(space['end'])
            number_of_slots_in_this_space = (
                end_date - start_date).seconds / 60 / 5
            i = 0

            if self.number_of_slots < number_of_slots_in_this_space:
                while i < (number_of_slots_in_this_space -
                           self.number_of_slots):
                    timeslot = TimeSlot(
                        start_date, start_date + self.number_of_slots * self.slot_delta)
                    result.append(timeslot)
                    i += 1
                    start_date = start_date + self.slot_delta

        return result

    def has_next(self):
        return (self.__iterator < len(self.__time_slots)) or self.are_more_slots

    def next(self):
        result = None

        if self.__iterator < len(self.__time_slots):
            result = self.__time_slots[self.__iterator]
            self.__iterator += 1
        elif self.are_more_slots:
            self.__time_slots = self.__filter()
            result = self.next()

        return result
