import copy
import math
import random
import sys

import pensa as pn


class WormholesMiller(pn.Agent):
    """
    An agent which attempts to end the game as quickly as possible by milling the draw pile
    """
    def __init__(self, name='Miller', verbose=False):
        super().__init__(name, verbose)
        self.game = None
        self.path = []

    def give_game(self, game):
        self.game = game

    def choose_move(self, moves):
        assert self.game.board is not None
        player_index = self.game.players.index(self)

        def is_goal_node(n):
            return any(t['type'] == 'P' for t, _ in self.game.board['map'].get_neighbors(n))

        def get_neighbors(n):
            map_moves = self.game.get_map_moves(n, self, self.game.board['wormholes'])
            return [(float('fuel' in map_move['text']), map_move['position']) for map_move in map_moves]

        if 'Start' in moves[0]['text']:
            _, found_path = pn.a_star(
                [move['position'] for move in moves],
                is_goal_node,
                get_neighbors,
            )
            self.path = found_path
        if not len(self.path):
            for index, move in enumerate(moves):
                if 'points' in move['text'] or 'Discard' in move['text'] or 'Pickup' in move['text']:
                    return index
            return 0
        for index, move in enumerate(moves):
            if ('Move to' in move['text'] or 'Start' in move['text']) and self.path[-1] == move['position']:
                self.path.pop()
                return index
        return 0


class WormholesHamilton(pn.Agent):
    """
    An agent which visits all planets in order ad infinitum
    Discards all passengers except those whose destinations are in the next d planets
    """
    def __init__(self, name=None, verbose=False, d=2):
        super().__init__(f'Hamilton {d}' if name is None else name, verbose)
        self.d = d
        self.game = None
        self.planet_num = 0
        self.path = []
        self.visiting_planet = False

    def give_game(self, game):
        self.game = game

    def accepted_passengers(self):
        return [(self.planet_num + m) % len(self.game.board['planets']) for m in range(self.d)]

    def choose_move(self, moves):
        assert self.game.board is not None
        player_index = self.game.players.index(self)
        player_data = self.game.board['player_data'][player_index]

        def is_goal_node(n):
            ns = self.game.board['map'].get_neighbors(n)
            return any(t['type'] == 'P' and t['id'] == self.planet_num for t, _ in ns)

        def get_neighbors(n):
            map_moves = self.game.get_map_moves(n, self, self.game.board['wormholes'])
            return [(float('fuel' in map_move['text']), map_move['position']) for map_move in map_moves]

        def heuristic(n):
            return 1.0 / (1.0 + len(self.game.get_map_moves(n, self, self.game.board['wormholes'])))

        if self.visiting_planet:
            for index, move in enumerate(moves):
                if 'Place wormhole' in move['text'] or 'Drop-off' in move['text'] or 'pickup' in move['text']:
                    return index
                if 'Discard passengers' in move['text'] and move['planet_id'] not in self.accepted_passengers():
                    return index
                if 'Pickup' in move['text']:
                    return index
            if not any(p in player_data['hand'].cards for p in self.accepted_passengers()):
                return 0
            else:
                self.visiting_planet = False
        if 'Start' in moves[0]['text']:
            _, found_path = pn.a_star(
                [move['position'] for move in moves],
                is_goal_node,
                get_neighbors,
                heuristic,
            )
            self.path = found_path
            print(self.path)
        if len(self.path) == 0 and not self.visiting_planet:
            _, next_path = pn.a_star(
                [player_data['position']],
                is_goal_node,
                get_neighbors,
                heuristic,
            )
            self.path = next_path[:-1]
            print(self.path)
        for index, move in enumerate(moves):
            if ('Move to' in move['text'] or 'Start' in move['text']) and self.path[-1] == move['position']:
                self.path.pop()
                if len(self.path) == 0:
                    self.visiting_planet = True
                    self.planet_num = (self.planet_num + 1) % len(self.game.board['planets'])
                return index
        return 0


class WormholesNaomi(pn.Agent):
    """
    An agent which plans 1 planet ahead, visiting whichever planet yields the most points per unit of fuel
    """
    def __init__(self, name=None, verbose=False):
        super().__init__(f'Naomi' if name is None else name, verbose)
        self.game = None
        self.total_wormholes = None
        self.path = []
        self.recalculate = False
        self.imp_planets = []

    def give_game(self, game):
        self.game = game

    def plan(self, start_positions):
        player_index = self.game.players.index(self)
        player_data = self.game.board['player_data'][player_index]
        num_planets = len(self.game.board['planets'])

        def is_goal_node_func(p):
            return lambda n: any(t['type'] == 'P' and t['id'] == p for t, _ in self.game.board['map'].get_neighbors(n))

        def get_neighbors(n):
            map_moves = self.game.get_map_moves(n, self, self.game.board['wormholes'])
            return [(float('fuel' in map_move['text']), map_move['position']) for map_move in map_moves]

        def heuristic(n):
            return 1.0 / (1.0 + len(self.game.get_map_moves(n, self, self.game.board['wormholes'])))

        planet_costs_paths = pn.dijkstra(
            start_positions,
            [is_goal_node_func(p) for p in range(num_planets)],
            get_neighbors,
            heuristic
        )
        self.imp_planets = []
        for p in range(num_planets):
            if planet_costs_paths[p][1] is not None:
                planet_costs_paths[p][1].insert(0, 'visit')
            else:
                self.imp_planets.append(p)

        connection_reward = 3.0 if (len(self.game.board['connected_planets']) >= 6) else 1.0
        planet_profits = [connection_reward for _ in self.game.board['planets']]
        for p in self.game.board['connected_planets']:
            planet_profits[p] = 0.0
        # no connection reward if another player will get there first
        if len(self.game.board['connected_planets']) < len(self.game.board['planets']):
            for index, player in enumerate(self.game.players):
                this_player_data = self.game.board['player_data'][index]
                if index == player_index or 'position' not in this_player_data:
                    continue
                found_costs_paths = pn.dijkstra(
                    [this_player_data['position']],
                    [is_goal_node_func(p) for p in range(len(self.game.board['planets']))],
                    get_neighbors,
                    heuristic,
                )
                for p, (found_cost, _) in enumerate(found_costs_paths):
                    if (not math.isfinite(found_cost)) or not math.isfinite(planet_costs_paths[p][0]):
                        if not planet_costs_paths[p][0] > found_cost:
                            planet_profits[p] = 0.0
                            continue
                    if math.ceil(found_cost / 3.0) <= math.ceil((planet_costs_paths[p][0] - player_data['fuel']) / 3.0):
                        planet_profits[p] = 0.0
        for passenger in player_data['hand'].cards:
            planet_profits[passenger] += 2.0
        if len(player_data['visited_planets']) >= 5:
            for planet_id in range(len(planet_profits)):
                if planet_id in player_data['hand'] and planet_id not in player_data['visited_planets']:
                    planet_profits[planet_id] += 3.0

        largest_roi = -1
        for profit, (cost, path) in zip(planet_profits, planet_costs_paths):
            if path is not None and len(path) > (2 if 'position' in player_data else 1) and profit / (
                    cost + .05) > largest_roi:
                largest_roi = profit / (cost + .05)
                self.path = path

        if 'position' in player_data:
            self.path.pop()
        print(self.path)
        if not len(self.path) > 1:
            print(planet_profits)
            print(planet_costs_paths)
        assert len(self.path) > 1

    def choose_move(self, moves):
        assert self.game is not None

        if self.total_wormholes is None:  # First turn
            self.total_wormholes = self.game.board['total_wormholes']
            self.plan([move['position'] for move in moves])

        player_index = self.game.players.index(self)
        player_data = self.game.board['player_data'][player_index]
        desired_move = self.path[-1]

        if 'visit' in desired_move:
            place_move, _ = next(filter(lambda en_move: 'Place' in en_move[1]['text'], enumerate(moves)), (None, None))
            if place_move is not None:
                return place_move
            drop_move, _ = next(filter(lambda en_move: 'Drop-off' in en_move[1]['text'], enumerate(moves)),
                                (None, None))
            if drop_move is not None:
                return drop_move
            pickup_move, _ = next(filter(lambda en_move: 'pickup' in en_move[1]['text'], enumerate(moves)),
                                  (None, None))
            if pickup_move is not None:
                return pickup_move
            discard_move, _ = next(filter(
                lambda en_move: 'Discard' in en_move[1]['text'] and int(en_move[1]['text'][-1]) in self.imp_planets,
                enumerate(moves)), (None, None))
            if discard_move is not None:
                return discard_move
            last_pickup_move, _ = next(filter(lambda en_move: 'Pickup' in en_move[1]['text'], enumerate(moves)),
                                       (None, None))
            if last_pickup_move is not None:
                return last_pickup_move
            if len(player_data['hand']) == 0:
                return 0
            else:
                self.path.pop()
                self.recalculate = True

        if self.recalculate or self.game.board['total_wormholes'] != self.total_wormholes:
            self.total_wormholes = self.game.board['total_wormholes']
            self.plan([player_data['position']])
            self.recalculate = False
            desired_move = self.path[-1]

        for index, move in enumerate(moves):
            if ('Move to' in move['text'] or 'Start' in move['text']) and desired_move == move['position']:
                self.path.pop()
                return index
        if player_data['fuel'] != 0 and desired_move != player_data['last_position']:
            print(f'Invalid move planned: {desired_move}')
            self.recalculate = True
            self.choose_move(moves)
        return 0


class WormholesFilip(pn.Agent):
    """
    An agent which plans d planets ahead, following whichever complete path yields the most points per unit fuel
    """
    def __init__(self, name=None, verbose=False, d=3):
        super().__init__(f'Filip' if name is None else name, verbose)
        assert 0 < d
        self.d = d
        self.game = None
        self.total_wormholes = None
        self.path = []
        self.recalculate = False
        self.imp_planets = []

    def give_game(self, game):
        self.game = game

    def plan(self, start_positions):
        player_index = self.game.players.index(self)
        player_data = self.game.board['player_data'][player_index]
        num_planets = len(self.game.board['planets'])

        def is_goal_node_func(p):
            return lambda n: any(t['type'] == 'P' and t['id'] == p for t, _ in self.game.board['map'].get_neighbors(n))

        all_goal_node_funcs = [is_goal_node_func(p) for p in range(num_planets)]
        all_goal_node_funcs.append(lambda n: any(t['type'] == 'B' for t, _ in self.game.board['map'].get_neighbors(n)))

        def get_neighbors_func(wormholes):
            return lambda node: [(float('fuel' in map_move['text']), map_move['position'])
                                 for map_move in self.game.get_map_moves(node, self, wormholes)]

        def heuristic(node):
            return 1.0 / (1.0 + len(self.game.get_map_moves(node, self, self.game.board['wormholes'])))

        # least turns any other player can take to reach a planet
        # TODO: recalculate as wormholes are placed by agent
        connection_distances = [self.game.board['map'].max_distance() for _ in self.game.board['planets']]
        if len(self.game.board['connected_planets']) < num_planets:
            for index, player in enumerate(self.game.players):
                this_player_data = self.game.board['player_data'][index]
                if index == player_index or 'position' not in this_player_data:
                    continue
                _found_costs_paths = pn.dijkstra(
                    [this_player_data['position']],
                    all_goal_node_funcs,
                    get_neighbors_func(self.game.board['wormholes']),
                    heuristic,
                )
                for p, (_found_cost, _) in enumerate(_found_costs_paths):
                    if p == num_planets:  # black hole
                        continue
                    connection_distances[p] = min(connection_distances[p], _found_cost)
        connection_distances = [math.ceil(x / 3.0) for x in connection_distances]

        def search_branch(
                depth,
                curr_positions,
                connected_planets,
                visited_planets,
                wormholes,
                hand,
                cost_since_pickup,
                prev_cost,
                prev_reward,
                prev_path
        ):
            assert len(prev_path) == 0 or isinstance(prev_path[0], str), f'expected prev_path {prev_path} to begin ' \
                                                                         f'with a string'

            def expand_branch(planet_index, found_cost, found_path):
                new_reward = 0

                # reward from planet connection
                wormholes_ = copy.deepcopy(wormholes)
                connected_planets_ = copy.deepcopy(connected_planets)
                if 'wormhole_id' not in self.game.board['map'][found_path[0]]:
                    player_wormholes = sorted(
                        filter(lambda witem: witem[0][0] == f'{player_index}', wormholes_.items()),
                        key=lambda witem: int(witem[0][1]),
                        reverse=True
                    )
                    if len(player_wormholes) == 0:
                        next_wormhole_num = 0
                    else:
                        last_player_wormhole = player_wormholes[0]
                        next_wormhole_num = int(last_player_wormhole[0][1]) + int(len(last_player_wormhole[1]) > 1)
                    if next_wormhole_num < self.game.max_wormholes:
                        # place a wormhole
                        next_wid = f'{player_index}{next_wormhole_num}'
                        if next_wid in wormholes_:
                            wormholes_[next_wid].append(found_path[0])
                        else:
                            wormholes_[next_wid] = [found_path[0]]
                        if planet_index not in connected_planets_ and \
                                math.ceil((found_cost + prev_cost - player_data['fuel']) / 3.0) \
                                <= connection_distances[planet_index]:
                            new_reward += 3.0 if (len(connected_planets_) >= 6) else 1.0
                            connected_planets_.append(planet_index)

                # reward from passenger drop-off
                new_hand = []
                for card in hand:
                    if card == -1:
                        new_reward += 2.0 / (num_planets - 1)
                    if card == planet_index:
                        new_reward += 2
                    else:
                        new_hand.append(card)
                if cost_since_pickup >= 3:
                    while len(new_hand) < 4:
                        new_hand.append(-1)
                elif len(new_hand) == 0:
                    found_cost += 3

                # reward from planet visit
                visited_planets_ = copy.deepcopy(visited_planets)
                if planet_index not in visited_planets_ and (planet_index in hand):
                    visited_planets_.add(planet_index)
                    if len(visited_planets) >= 5:
                        new_reward += 3.0

                if depth - 1 > 0:
                    new_roi, new_path = search_branch(
                        depth - 1,
                        [found_path[0]],
                        connected_planets_,
                        visited_planets_,
                        wormholes_,
                        new_hand,
                        found_cost + (cost_since_pickup if cost_since_pickup < 3 else 0),
                        found_cost + prev_cost,
                        new_reward + prev_reward,
                        ['visit'] + (found_path[:-1] if len(curr_positions) == 1 else found_path) + prev_path,
                    )
                else:
                    new_roi = (new_reward + prev_reward) / (found_cost + prev_cost)
                    new_path = ['visit'] + (found_path[:-1] if len(curr_positions) == 1 else found_path) + prev_path
                return new_roi, new_path

            best_roi, best_path = -1, None
            found_costs_paths = pn.dijkstra(
                curr_positions,
                all_goal_node_funcs,
                get_neighbors_func(wormholes),
                heuristic
            )
            for planet_index, (found_cost, found_path) in enumerate(found_costs_paths):
                if found_path is None:
                    continue
                if planet_index == num_planets:
                    if 'rounds_remaining' in self.game.board and \
                            math.ceil((1 + found_cost + prev_cost - player_data['fuel']) / 3.0) > \
                            self.game.board['rounds_remaining'] - 1:
                        # cannot reach the end of the branch
                        continue
                    if len(self.game.board['draw_pile']) == 0:
                        continue
                    cum_roi = 0.0
                    for drawn_planet in range(num_planets):
                        if drawn_planet in self.game.board['draw_pile']:
                            end_positions = set()
                            for planet_position in self.game.board['planets'][drawn_planet]:
                                for adj_tile, adj_position in self.game.board['map'].get_neighbors(planet_position):
                                    if adj_tile['type'] not in ' P':
                                        end_positions.add(adj_position)
                            end_position = random.choice(list(end_positions))
                            new_roi, _ = expand_branch(drawn_planet, 1 + found_cost, [end_position])
                            cum_roi += new_roi * self.game.board['draw_pile'].count(drawn_planet)
                    cum_roi /= len(self.game.board['draw_pile'])
                    if cum_roi > best_roi:
                        best_roi = cum_roi
                        best_path = ['visit', 'blackhole maneuver'] + (
                            found_path[:-1] if len(curr_positions) == 1 else found_path
                        ) + prev_path
                    continue
                if len(found_path) == 1:
                    continue
                if 'rounds_remaining' in self.game.board and \
                        math.ceil((found_cost + prev_cost - player_data['fuel']) / 3.0) > \
                        self.game.board['rounds_remaining'] - 1:
                    # cannot reach the end of the branch
                    continue
                new_roi, new_path = expand_branch(planet_index, found_cost, found_path)
                if new_roi > best_roi:
                    best_roi, best_path = new_roi, new_path
            if best_path is None:
                return best_roi, prev_path
            else:
                return best_roi, best_path

        self.imp_planets = []
        for (_, p) in pn.dijkstra(
                start_positions,
                all_goal_node_funcs,
                get_neighbors_func(self.game.board['wormholes']),
                heuristic
        ):
            if p is None:
                self.imp_planets.append(p)

        # how to work with starting move?
        _, self.path = search_branch(
            self.d,
            start_positions,
            connected_planets=self.game.board['connected_planets'],
            visited_planets=player_data['visited_planets'],
            wormholes=self.game.board['wormholes'],
            hand=player_data['hand'].cards,
            cost_since_pickup=3 - (player_data['fuel'] if player_data['has_boarded'] else 0),
            prev_cost=0.01,
            prev_reward=0,
            prev_path=[]
        )
        '''if 'position' in player_data and len(self.path) > 0:
            self.path.pop()'''
        print(self.path)

    def choose_move(self, moves):
        assert self.game is not None

        if self.total_wormholes is None:  # First turn
            self.total_wormholes = self.game.board['total_wormholes']
            self.plan([move['position'] for move in moves])
        if len(self.path) == 0:
            print(f'{self.name} could not find any move')
            return 0

        player_index = self.game.players.index(self)
        player_data = self.game.board['player_data'][player_index]
        desired_move = self.path[-1]

        if 'blackhole maneuver' == desired_move:
            if player_data['fuel'] == 0:
                self.recalculate = True
                return 0
            texts = [move['text'] for move in moves]
            if 'Perform black hole maneuver [-1 fuel]' not in texts:
                print(f'Invalid move planned: {desired_move}')
                self.recalculate = True
                self.choose_move(moves)
            else:
                self.path.pop()
                return texts.index('Perform black hole maneuver [-1 fuel]')

        if 'visit' == desired_move:
            place_move, _ = next(filter(lambda en_move: 'Place' in en_move[1]['text'], enumerate(moves)), (None, None))
            if place_move is not None:
                return place_move
            drop_move, _ = next(filter(lambda en_move: 'Drop-off' in en_move[1]['text'], enumerate(moves)),
                                (None, None))
            if drop_move is not None:
                return drop_move
            pickup_move, _ = next(filter(lambda en_move: 'pickup' in en_move[1]['text'], enumerate(moves)),
                                  (None, None))
            if pickup_move is not None:
                return pickup_move
            discard_move, _ = next(filter(
                lambda en_move: 'Discard' in en_move[1]['text'] and int(en_move[1]['text'][-1]) in self.imp_planets,
                enumerate(moves)), (None, None))
            if discard_move is not None:
                return discard_move
            last_pickup_move, _ = next(filter(lambda en_move: 'Pickup' in en_move[1]['text'], enumerate(moves)),
                                       (None, None))
            if last_pickup_move is not None:
                return last_pickup_move
            if len(player_data['hand']) == 0:
                return 0
            else:
                self.path.pop()
                self.recalculate = True

        if self.recalculate or self.game.board['total_wormholes'] != self.total_wormholes:
            self.total_wormholes = self.game.board['total_wormholes']
            self.recalculate = False
            self.plan([player_data['position']])
            if len(self.path) == 0:
                print(f'{self.name} could not find any move')
                return 0
            desired_move = self.path[-1]

        for index, move in enumerate(moves):
            if ('Move to' in move['text'] or 'Start' in move['text']) and desired_move == move['position']:
                self.path.pop()
                return index
        if player_data['fuel'] != 0 and desired_move != player_data['last_position']:
            print(f'Invalid move planned: {desired_move}')
            self.recalculate = True
            self.choose_move(moves)
        return 0


class Sector:
    def __init__(self, s):
        ss = [*s]
        self.tiles = list()
        self.planet_tiles = list()
        for dx, dy in sector_deltas:
            c = ss.pop(0)
            if c == 'P':
                self.planet_tiles.append((dx, dy))
            else:
                self.tiles.append((dx, dy, c))

    def is_invalid(self, tile_map, x, y):
        for px, py in self.planet_tiles:
            for dx, dy in sector_deltas:
                if (px + dx, py + dy) not in sector_deltas:
                    position = (x + px + dx, y + py + dy)
                    if 0 <= position[0] < tile_map.cols and 0 <= position[1] < tile_map.rows:
                        if tile_map[position]['type'] in ['P', 'S']:
                            return True
        return False


class WormholesGame(pn.Boardgame):
    """
    Implementation of the eponymous 2022 board game by Peter McPherson
    """
    def __init__(self, players, seed=None):
        assert 1 <= len(players) <= 5
        if seed is None:
            seed = random.randrange(sys.maxsize)
        print(f"random seed: {seed}")
        random.seed(seed)
        self.max_wormholes = 5
        self.max_hand_size = 4
        # TODO: implement sector rotation and guarantee no duplicate sectors
        if len(players) < 4:
            map_template = random.choice(small_map_templates)
            satellite_sector = Sector('####O#OO#S#OO#O####')
        else:
            map_template = random.choice(large_map_templates)
            satellite_sector = Sector('#O#OO#SOOS#SOOO#O##')
        print(map_template['name'])
        board = {
            'map': pn.HexGrid(map_template['rows'], map_template['cols'], ' ', self.stringify),
            'planets': [],  # list of lists of positions
            'connected_planets': [],  # list of planet ids
            'orbits': [[]],  # list of lists of positions
            'wormholes': dict(),  # dictionary of lists of positions
            'total_wormholes': 0,
            'player_data': [{
                'fuel': 3,
                'hand': pn.Deck(),
                'has_boarded': False,
                'last_position': None,
                'score': 0.0,
                'visited_planets': set(),
                'wormhole_num': 0,
            } for _ in players],
            'draw_pile': pn.Deck(),
            'discard_pile': pn.Deck(),
        }
        for dx, dy, tile_type in satellite_sector.tiles:
            x, y = map_template['satellite_position']
            if tile_type == 'O':
                board['orbits'][0].append((x + dx, y + dy))
                board['map'][(x + dx, y + dy)] = {'type': tile_type, 'id': 0}
            else:
                board['map'][(x + dx, y + dy)] = {'type': tile_type}
        for x, y in map_template['sector_positions']:
            sector = random.choice(sectors)
            while sector.is_invalid(board['map'], x, y):
                print('invalid sector found, resampling ...')
                sector = random.choice(sectors)
            orbit_id = None
            for dx, dy, tile_type in sector.tiles:
                if tile_type in '0123456789':
                    wormhole_id = 'W' + tile_type
                    board['map'][(x + dx, y + dy)] = {'type': '#', 'wormhole_id': wormhole_id}
                    if wormhole_id not in board['wormholes']:
                        board['wormholes'][wormhole_id] = [(x + dx, y + dy)]
                    else:
                        board['wormholes'][wormhole_id].append((x + dx, y + dy))
                    board['total_wormholes'] += 1
                elif tile_type == 'O':
                    if orbit_id is None:
                        orbit_id = len(board['orbits'])
                        board['orbits'].append(list())
                    board['orbits'][orbit_id].append((x + dx, y + dy))
                    board['map'][(x + dx, y + dy)] = {'type': tile_type, 'id': orbit_id}
                else:
                    board['map'][(x + dx, y + dy)] = {'type': tile_type}
            if len(sector.planet_tiles) > 2:
                planet_id = len(board['planets'])
                board['planets'].append(list())
                for dx, dy in sector.planet_tiles:
                    board['map'][(x + dx, y + dy)] = {'type': 'P', 'id': planet_id}
                    board['planets'][planet_id].append((x + dx, y + dy))
            else:
                for dx, dy in sector.planet_tiles:
                    planet_id = len(board['planets'])
                    board['planets'].append([(x + dx, y + dy)])
                    board['map'][(x + dx, y + dy)] = {'type': 'P', 'id': planet_id}
        # each planet gets a number of cards equal to the total number of planets
        board['draw_pile'].add_all(range(len(board['planets'])), len(board['planets']))
        board['player_data'][0]['hand'].add(board['draw_pile'].draw())
        for index in range(1, len(players)):
            board['player_data'][index]['hand'].add_all(board['draw_pile'].draw(2))
        super().__init__(players, board)

    def stringify(self, tile):
        """stringification key
            # is an empty tile
            S is a station tile
            O is an orbit tile
            P is a planet tile

            B is a black hole
            N is a nebula tile
            X is a slingshot tile

            UL: A if player(s) present
            UM: base tile type
            UR: tile identifier
            BL: W if wormhole present
            BM: owning player identifier
            BR: wormhole identifier
        """
        out = list()
        if 'players' in tile and len(tile['players']) > 0:
            out.append('A')
        else:
            out.append(tile['type'])
        if 'id' in tile:
            strid = str(tile['id'])
            if len(strid) == 1:
                out[0] += tile['type'] + strid
            else:
                out[0] += strid[-2:]
        else:
            out[0] += tile['type'] * 2
        if 'wormhole_id' in tile:
            out.append('W' + tile['wormhole_id'])
        else:
            out.append(tile['type'] * 3)
        return out

    def finished(self):
        return self.board.get('rounds_remaining', -1) == 0

    def get_map_moves(self, current_position, player, wormholes, exclude=None):
        player_index = self.players.index(player)
        player_data = self.board['player_data'][player_index]
        current_tile = self.board['map'][current_position]
        neighbors = self.board['map'].get_neighbors(current_position)
        map_moves = []
        if exclude is None:
            movement = set()
        else:
            movement = set(exclude)
        movement.add(current_position)

        if 'wormhole_id' in current_tile:
            wormhole_id = current_tile['wormhole_id']
            destinations = []
            destinations.extend(wormholes[wormhole_id])
            if wormhole_id[0] == 'W':
                player_wormholes = wormholes.get(str(player_index) + wormhole_id[1], [])
                if len(player_wormholes) > 1:
                    destinations.extend(player_wormholes)
                else:
                    destinations = []
            elif len(destinations) > 1:
                destinations.extend(wormholes.get('W' + wormhole_id[1], []))
            for destination in destinations:
                if destination not in movement:
                    map_moves.append({
                        'text': f'Move to {destination} through wormhole {wormhole_id}',
                        'position': destination,
                        'wormhole_id': wormhole_id
                    })
                    movement.add(destination)
        if current_tile['type'] == 'N':
            for t, tp in neighbors:
                if t['type'] in '#ONX' and tp not in movement:
                    map_moves.append({
                        'text': f'Move to {tp}',
                        'position': tp,
                    })
                    movement.add(tp)
        if current_tile['type'] == 'X':
            for dx, dy in [(1, -1), (-1, 1), (1, 1), (-1, -1), (0, 2), (0, -2)]:
                x = current_position[0] + dx
                y = current_position[1] + dy
                while 0 <= x < self.board['map'].cols and 0 <= y < self.board['map'].rows:
                    t, tp = self.board['map'][(x, y)], (x, y)
                    if t['type'] in '#ONX' and tp not in movement:
                        map_moves.append({
                            'text': f'Move to {tp} via starsling [-1 fuel]',
                            'position': tp,
                        })
                        movement.add(tp)
                    x += dx
                    y += dy
        elif current_tile['type'] == 'O':
            for tp in self.board['orbits'][current_tile['id']]:
                if tp not in movement:
                    map_moves.append({
                        'text': f'Move to {tp} via orbit [-1 fuel]',
                        'position': tp,
                    })
                    movement.add(tp)
        for t, tp in neighbors:
            if t['type'] in '#ONX' and tp not in movement:
                map_moves.append({
                    'text': f'Move to {tp} [-1 fuel]',
                    'position': tp,
                })
                movement.add(tp)
        return map_moves

    def get_moves(self, player):
        moves = list()
        player_index = self.players.index(player)
        player_data = self.board['player_data'][player_index]
        if 'position' in player_data:
            current_tile = self.board['map'][player_data['position']]
            neighbors = self.board['map'].get_neighbors(player_data['position'])
        else:
            current_tile = None
            neighbors = None
        if current_tile is None:
            for tp in self.board['orbits'][0]:
                if 'wormhole_id' not in self.board['map'][tp]:
                    moves.append({'text': f'Start at {tp}', 'position': tp})
        # TODO: choose from discard pile
        elif 'pickup_in_progress' in player_data:
            for planet_id in set(player_data['hand'].cards):
                moves.append({
                    'text': f'Discard passengers with destination {planet_id}',
                    'planet_id': planet_id,
                })
            moves.append(player_data['pickup_in_progress'])
        else:
            moves.append({'text': 'End turn'})
            if player_data['fuel'] > 0:
                moves.extend(self.get_map_moves(player_data['position'], player, self.board['wormholes'],
                                                [player_data['last_position']]))
                if len(self.board['draw_pile']) and any(t['type'] == 'B' for t, _ in neighbors):
                    moves.append({'text': 'Perform black hole maneuver [-1 fuel]'})
            else:
                map_moves = self.get_map_moves(player_data['position'], player, self.board['wormholes'],
                                               [player_data['last_position']])
                moves.extend([move for move in map_moves if 'fuel' not in move['text']])
            # TODO: allow placing wormholes adjacent to ship
            # TODO: disallow multiple wormholes adjacent to the same planet from the same player
            if ('wormhole_id' not in current_tile) and (player_data['wormhole_num'] < 2 * self.max_wormholes):
                next_wormhole_id = str(player_index) + str(player_data['wormhole_num'] // 2)
                new_move = {
                    'text': f"Place wormhole {next_wormhole_id}",
                    'wormhole_id': next_wormhole_id,
                }
                for t, _ in neighbors:
                    if t['type'] == 'P' and t['id'] not in self.board['connected_planets']:
                        reward = 3.0 if (len(self.board['connected_planets']) >= 6) else 1.0
                        new_move['text'] += f" [+{reward} points]"
                        new_move['reward'] = reward
                        new_move['connected_planet'] = t['id']
                        break
                moves.append(new_move)
            for t, _ in neighbors:
                if t['type'] == 'P':
                    if t['id'] in player_data['hand']:
                        reward = 2 * player_data['hand'].count(t['id'])
                        if len(player_data['visited_planets']) >= 5 and t['id'] not in player_data['visited_planets']:
                            reward += 3.0
                        moves.append({
                            'text': f"Drop-off passengers at planet {t['id']} [+{reward} points]",
                            'planet_id': t['id'],
                            'reward': reward,
                        })
                    if not player_data['has_boarded']:
                        if len(player_data['hand']) > 0:
                            moves.append({
                                'text': f"Discard/pickup passengers from planet {t['id']}",
                                'planet_id': t['id'],
                            })
                        else:
                            moves.append({'text': f"Pickup passengers from planet {t['id']}", 'planet_id': t['id']})
                    break
                if t['type'] == 'S':
                    if (not player_data['has_boarded']) and len(self.board['discard_pile']):
                        if len(player_data['hand']) > 0:
                            moves.append({'text': 'Discard/pickup passengers from station'})
                        else:
                            moves.append({'text': 'Pickup passengers from station'})
                    break
        return moves

    def do_move(self, player, move, verbose=False):
        player_index = self.players.index(player)
        player_data = self.board['player_data'][player_index]
        player_data['last_position'] = None
        turn_end = False
        if 'position' in player_data:
            current_tile = self.board['map'][player_data['position']]
        else:
            current_tile = None

        if 'End turn' in move['text']:
            turn_end = True
        elif 'Start' in move['text']:
            player_data['position'] = move['position']
            start_tile = self.board['map'][move['position']]
            if 'players' not in start_tile:
                start_tile['players'] = []
            start_tile['players'].append(player_index)
            player_data['wormhole_num'] = 1
            wormhole_id = str(player_index) + '0'
            start_tile['wormhole_id'] = wormhole_id
            self.board['wormholes'][wormhole_id] = [move['position']]
            self.board['total_wormholes'] += 1
        elif 'Move to' in move['text']:
            if '-1 fuel' in move['text']:
                player_data['fuel'] -= 1
            player_data['last_position'] = player_data['position']
            current_tile['players'].remove(player_index)
            player_data['position'] = move['position']
            end_tile = self.board['map'][move['position']]
            if 'players' not in end_tile:
                end_tile['players'] = []
            end_tile['players'].append(player_index)
            if 'wormhole_id' in move:
                owner_id = move['wormhole_id'][0]
                if owner_id != 'W' and int(owner_id) != player_index:
                    self.board['player_data'][int(owner_id)]['score'] += 1
        elif 'black hole maneuver' in move['text']:
            player_data['fuel'] -= 1
            player_data['last_position'] = player_data['position']
            current_tile['players'].remove(player_index)
            drawn_planet = self.board['draw_pile'].draw()
            self.board['discard_pile'].add(drawn_planet)
            end_positions = set()
            for planet_position in self.board['planets'][drawn_planet]:
                for adj_tile, adj_position in self.board['map'].get_neighbors(planet_position):
                    if adj_tile['type'] not in ' P':
                        end_positions.add(adj_position)
            end_position = random.choice(list(end_positions))
            end_tile = self.board['map'][end_position]
            player_data['position'] = end_position
            if 'players' not in end_tile:
                end_tile['players'] = []
            end_tile['players'].append(player_index)
        elif 'Place wormhole' in move['text']:
            player_data['wormhole_num'] += 1
            current_tile['wormhole_id'] = move['wormhole_id']
            if move['wormhole_id'] in self.board['wormholes']:
                self.board['wormholes'][move['wormhole_id']].append(player_data['position'])
            else:
                self.board['wormholes'][move['wormhole_id']] = [player_data['position']]
            self.board['total_wormholes'] += 1
            if 'reward' in move:
                player_data['score'] += move['reward']
            if 'connected_planet' in move:
                self.board['connected_planets'].append(move['connected_planet'])
                if len(self.board['connected_planets']) == len(self.board['planets']):
                    self.board['rounds_remaining'] = 4
        elif 'Drop-off passengers at planet' in move['text']:
            player_data['hand'].take(move['planet_id'])
            player_data['score'] += move['reward']
            player_data['visited_planets'].add(move['planet_id'])
        elif 'Discard/pickup passengers from' in move['text']:
            move['text'] = 'P' + move['text'][9:]
            player_data['pickup_in_progress'] = move
        elif 'Discard passengers with destination' in move['text']:
            self.board['discard_pile'].add_all(player_data['hand'].take(move['planet_id']))
            if len(player_data['hand']) == 0:  # pickup will be the only available move
                if verbose:
                    print(f"{player.name}: {player_data['pickup_in_progress']['text']}")
                self.do_move(player, player_data['pickup_in_progress'], verbose=verbose)
                turn_end = True
        elif 'Pickup passengers from' in move['text']:
            if 'pickup_in_progress' in player_data:
                player_data.pop('pickup_in_progress')
            while len(player_data['hand']) < self.max_hand_size:
                if 'station' in move['text']:
                    draw = self.board['discard_pile'].draw()
                    if draw is None:
                        break
                    else:
                        player_data['hand'].add(draw)
                else:
                    draw = self.board['draw_pile'].draw()
                    if draw is None:
                        self.board['rounds_remaining'] = min(2, self.board.get('rounds_remaining', 2))
                        break
                    else:
                        if draw == move.get('planet_id', -1):
                            self.board['discard_pile'].add(draw)
                        else:
                            player_data['hand'].add(draw)
            player_data['has_boarded'] = True
        else:
            raise NotImplementedError

        if not turn_end:
            moves = self.get_moves(player)
            if verbose and player.verbose:
                self.display_player(player)
            move = moves[player.choose_move(moves)]
            if verbose:
                print(f"{player.name}: {move['text']}")
            self.do_move(player, move, verbose=verbose)

    def end_round(self):
        if 'rounds_remaining' in self.board:
            self.board['rounds_remaining'] -= 1
        for player_index in range(len(self.players)):
            self.board['player_data'][player_index]['fuel'] = 3
            self.board['player_data'][player_index]['has_boarded'] = False

    def display(self):
        print(str(self.board['map']))
        for player in self.players:
            self.display_player(player)
        print(f"There are {len(self.board['draw_pile'])} cards in the draw pile")
        print(f"There are {len(self.board['discard_pile'])} cards in the discard pile")
        if 'rounds_remaining' in self.board:
            print(f"There are {self.board['rounds_remaining']} round(s) remaining!")
        print('')

    def display_player(self, player):
        player_index = self.players.index(player)
        player_data = self.board['player_data'][player_index]
        if 'position' in player_data:
            print(f"{player.name} is at {player_data['position']}")
        print(f"{player.name} has {player_data['fuel']} fuel and {player_data['score']} points")
        if len(player_data['hand']):
            print(f"their passengers are {player_data['hand'].cards}")

    def finish(self, verbose=False):
        if verbose:
            results = sorted(
                [(p.name, pd['score']) for p, pd in zip(self.players, self.board['player_data'])],
                key=lambda tup: -tup[1]
            )
            if results[0][1] > results[1][1]:
                print(f"\n{results[0][0]} is the winner!\n")
            else:
                print("\nIt's a tie!\n")
            print('     Player         Score')
            print('--------------------------')
            for n, s in results:
                print("{:<16}    {:5.1f}\n".format(n, s))


sector_deltas = list()
for i, j in enumerate([0, 1, 2, 1, 2, 1, 2, 1, 0]):
    for k in range(-j, j + 1, 2):
        sector_deltas.append((k, i - 4))
sectors = [Sector(s) for s in [
    'NN#### #####P##### ', '#########P# ## # ##', ' ## ###P#####X#### ', '##### # ###P### ###', ' #######P#P  ######',
    'NN#N#P## #### #### ', '# ##########P# # ##', 'P## ### # #  # #   ', '## ##### ###  ###P#', '###### #####P ##4# ',
    '# ### ###P#### ### ', '###########BP ## ##', '# ### #### P #  # #', 'NNN#####P## ## P###', '### O OOOP#POOP#OOO',
    '####2### # #P# ####', '   # ####P#### ### ', '## #######3 # ###P#',
]]
# 1-3 players: 8 sectors (including station) with 8 planets with 8 cards each
small_map_templates = [
    {'cols': 18, 'rows': 30, 'satellite_position': (5, 19), 'sector_positions': [
        (2, 12), (4, 4), (7, 11), (10, 18), (12, 10), (15, 17), (13, 25),
    ], 'name': 'standard small'},
    {'cols': 21, 'rows': 25, 'satellite_position': (7, 5), 'sector_positions': [
        (2, 6), (12, 4), (5, 13), (15, 11), (8, 20), (13, 19), (18, 18),
    ], 'name': 'Asteroidea I'},
    {'cols': 18, 'rows': 32, 'satellite_position': (5, 21), 'sector_positions': [
        (7, 13), (10, 20), (12, 12), (2, 14), (8, 28), (9, 5), (15, 19),
    ], 'name': 'Decapoda II'},
]
# 4-5 players: 10 sectors (including station) with 10 planets with 10 cards each
large_map_templates = [
    {'cols': 18, 'rows': 33, 'satellite_position': (8, 28), 'sector_positions': [
        (13, 27), (5, 21), (10, 20), (15, 19), (2, 14), (7, 13), (12, 12), (4, 6), (9, 5),
    ], 'name': 'standard large'},
    {'cols': 24, 'rows': 31, 'satellite_position': (5, 13), 'sector_positions': [
        (10, 12), (13, 19), (18, 18), (8, 20), (7, 5), (15, 11), (16, 26), (21, 25), (2, 6),
    ], 'name': 'Geotria VII'},
    {'cols': 20, 'rows': 33, 'satellite_position': (11, 11), 'sector_positions': [
        (9, 19), (14, 18), (17, 25), (12, 26), (7, 27), (4, 20), (6, 12), (8, 4), (2, 28),
    ], 'name': 'Chaetodon III'},
]

if __name__ == "__main__":
    players = [
        # UserAgent('User'),
        # WormholesHamilton(),
        WormholesNaomi(),
        # WormholesFilip(),
    ]
    # random.shuffle(players)
    game = WormholesGame(players=players, seed=None)
    game.simulate(verbose=True)
    # seed with inaccessible planets: 869615425253236833
