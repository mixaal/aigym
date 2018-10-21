class Recorder(object):

    def __init__(self, states=[]):
        self._states = states
        self._idx = len(states)

    def record(self, state):
        self._states.append(state)

    def rewind(self):
        self._idx = 0

    def has_playback(self):
        return self._idx < len(self._states)

    def play(self):
        s = self._states[self._idx]
        self._idx += 1
        return s

    def _get_states(self):
        return self._states
