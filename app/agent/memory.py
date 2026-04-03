class Memory:
    def __init__(self):
        self.store = {}

    def get(self, user_id):
        return self.store.get(user_id, [])

    def add(self, user_id, msg):
        self.store.setdefault(user_id, []).append(msg)