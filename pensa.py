import heapq
import random


class Boardgame:
    def __init__(self, players, board):
        self.players = players
        for player in players:
            method = getattr(player, 'give_game', None)
            if callable(method):
                method(self)
        self.board = board

    def simulate(self, verbose=False):
        while not self.finished():
            if verbose:
                self.display()
            for index, player in enumerate(self.players):
                moves = self.get_moves(player)
                if verbose and player.verbose and index > 0:
                    self.display_player(player)
                move = moves[player.choose_move(moves)]
                if verbose:
                    print(f"{player.name}: {move['text']}")
                self.do_move(player, move, verbose=verbose)
            self.end_round()
        self.finish(verbose=verbose)

    def finished(self):
        raise NotImplementedError('finished() call to un-extended Boardgame object')

    def get_moves(self, player):
        raise NotImplementedError('get_moves() call to un-extended Boardgame object')

    def do_move(self, player, move, verbose=False):
        raise NotImplementedError('do_move() call to un-extended Boardgame object')

    def end_round(self):
        raise NotImplementedError('end_round() call to un-extended Boardgame object')

    def display(self):
        raise NotImplementedError('display() call to un-extended Boardgame object')

    def display_player(self, player):
        raise NotImplementedError('display_player() call to un-extended Boardgame object')

    def finish(self, verbose=False):
        raise NotImplementedError('finish() call to un-extended Boardgame object')


class HexGrid:
    """
    Data structure for representing hexagonal grids
    Supports displaying the grid as text
    """
    def __init__(self, rows, cols, default, stringify_tile):
        self.rows = rows
        self.cols = cols
        self.grid = [[{'type': default} for _ in range(1 + rows // 2)] for _ in range(cols)]
        self.stringify_tile = stringify_tile

    def __getitem__(self, item):
        assert len(item) == 2, f'expected item {item} to be a tuple'
        x, y = item
        temp = self.grid[x]
        return temp[y // 2]

    def __setitem__(self, key, value):
        assert len(key) == 2
        x, y = key
        temp = self.grid[x]
        temp[y // 2] = value

    def max_distance(self):
        return (1 + self.rows + self.cols) // 2

    def get_neighbors(self, position):
        assert len(position) == 2
        x, y = position
        neighbors = list()
        for dx, dy in [(1, -1), (1, 1), (0, 2)]:
            if 0 <= x + dx < self.cols and 0 <= y + dy < self.rows:
                neighbors.append((self.__getitem__((x + dx, y + dy)), (x + dx, y + dy)))
            if 0 <= x - dx < self.cols and 0 <= y - dy < self.rows:
                neighbors.append((self.__getitem__((x - dx, y - dy)), (x - dx, y - dy)))
        return neighbors

    def __str__(self):
        first_tile = self.stringify_tile(self.grid[0][0])
        margin = round(4.8 * len(first_tile))
        spacing = 2 * margin - len(first_tile[0])
        out = '   | '
        for c in range(self.cols):
            out += '{:2d}'.format(c) + ' ' * (margin - 2)
        out += '\n---+-' + '-' * (margin * self.cols) + '\n'
        for y in range(self.rows):
            next_lines = ['   | ' for _ in first_tile]
            next_lines[0] = '{:2d} | '.format(y)
            if y % 2:
                next_lines = [s + ' ' * margin for s in next_lines]
            for x in range(y % 2, self.cols, 2):
                stringified = self.stringify_tile(self.grid[x][y // 2])
                next_lines = [s + t + ' ' * spacing for s, t in zip(next_lines, stringified)]
            for line in next_lines:
                out += line + '\n'
        return out


class Deck:
    """
    Data structure for representing a deck of cards (or other objects)
    """
    def __init__(self, iterable=None):
        if iterable is not None:
            self.cards = [*iterable]
        else:
            self.cards = []

    def __contains__(self, item):
        return self.cards.__contains__(item)

    def __len__(self):
        return self.cards.__len__()

    def count(self, value):
        return self.cards.count(value)

    def draw(self, num=None, default=None):
        if num is None:
            if len(self.cards):
                return self.cards.pop(random.randint(0, len(self.cards) - 1))
            else:
                return default
        else:
            draws = []
            while num > 0 and len(self.cards):
                index = random.randint(0, len(self.cards) - 1)
                draws.append(self.cards.pop(index))
                num -= 1
            if num > 0 and default is not None:
                draws.extend([default] * int(num))
            return draws

    def take(self, item, num=None):
        taken = []
        try:
            while num is None or num > 0:
                index = self.cards.index(item)
                taken.append(self.cards.pop(index))
                if num is not None:
                    num -= 1
        except ValueError:
            pass
        return taken

    def add(self, item, n=None):
        if n is None:
            self.cards.append(item)
        else:
            self.cards.extend([item] * n)

    def add_all(self, iterable, n=None):
        if n is None:
            self.cards.extend([*iterable])
        else:
            for item in iterable:
                self.cards.extend([item] * n)


def reconstruct_path(came_from, current_node):
    """Helper method for pathfinding algorithms"""
    path = [current_node]
    while came_from[path[-1]] is not None:
        path.append(came_from[path[-1]])
    return path


def a_star(start_nodes, is_goal_node, get_neighbors, h=lambda _: 0.0):
    """Pathfind from any node in start_nodes to any node for which is_goal_node evaluates to true"""
    open_heap = [(h(start_node), 0.0, None, start_node) for start_node in start_nodes]
    came_from = dict()
    while len(open_heap):
        est_total_cost, total_cost, prev_node, current_node = heapq.heappop(open_heap)
        if current_node in came_from:
            continue
        came_from[current_node] = prev_node
        if is_goal_node(current_node):
            return total_cost, reconstruct_path(came_from, current_node)
        for edge_cost, neighbor in get_neighbors(current_node):
            new_total_cost = total_cost + edge_cost
            if neighbor not in came_from:
                heapq.heappush(open_heap, (h(neighbor) + new_total_cost, new_total_cost, current_node, neighbor))
    return float('infinity'), None


def dijkstra(start_nodes, is_goal_nodes, get_neighbors, h=lambda _: 0.0):
    """Pathfind from any node in start_nodes to every type of goal node provided"""
    answers = [(float('infinity'), None) for _ in is_goal_nodes]
    open_heap = [(h(start_node), 0.0, None, start_node) for start_node in start_nodes]
    came_from = dict()
    while len(open_heap):
        est_total_cost, total_cost, prev_node, current_node = heapq.heappop(open_heap)
        if current_node in came_from:
            continue
        came_from[current_node] = prev_node
        for index, is_goal_node in enumerate(is_goal_nodes):
            if answers[index][1] is None and is_goal_node(current_node):
                answers[index] = (total_cost, reconstruct_path(came_from, current_node))
        if not any(path is None for _, path in answers):
            return answers
        for edge_cost, neighbor in get_neighbors(current_node):
            new_total_cost = total_cost + edge_cost
            if neighbor not in came_from:
                heapq.heappush(open_heap, (h(neighbor) + new_total_cost, new_total_cost, current_node, neighbor))
    return answers


class Agent:
    def __init__(self, name=None, verbose=False):
        self.name = input("Please type the (next) player's name: ") if name is None else name
        self.verbose = verbose

    def choose_move(self, moves):
        raise NotImplementedError('choose_move() call to un-extended Agent object')


class UserAgent(Agent):
    def __init__(self, name=None, verbose=True):
        super().__init__(name, verbose)

    def choose_move(self, moves):
        print(f'\n{self.name}\nPlease type the index of your move:')
        for index, move in enumerate(moves):
            print(f"({index}) {move['text']}")
        user_input = input('')
        while True:
            if user_input.isdecimal() and 0 <= int(user_input) < len(moves):
                return int(user_input)
            temp = [(index, m) for index, m in enumerate(moves) if user_input in m['text']]
            if len(temp) == 1:
                return temp[0][0]
            user_input = input('Input unrecognized, please retry:')


class BoredAgent(Agent):
    """
    Agent which first chooses the move it has previously taken the fewest times and
    second choose the move which provides the greatest immediate reward
    """
    def __init__(self, name=None, verbose=False):
        super().__init__(name, verbose)
        self.memory = dict()

    def choose_move(self, moves):
        best_moves = list()
        for move in moves:
            if not len(best_moves):
                best_moves.append(move)
                continue
            delta = self.memory.get(best_moves[0]['text'], 0) - self.memory.get(move['text'], 0)
            if delta == 0:
                delta = move.get('reward', 0) - best_moves[0].get('reward', 0)
            if delta > 0:
                best_moves = [move]
            elif delta == 0:
                best_moves.append(move)
        choice = random.choice(best_moves)
        self.memory[choice['text']] = self.memory.get(choice['text'], 0) + 1
        return moves.index(choice)
