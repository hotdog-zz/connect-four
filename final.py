from random import choice, choices
import numpy as np
import time

columns = 7
rows = 6

## 快速获取特殊的mask
def bottom_mask():
    ## bottom的mask
    temp = 0
    for i in range(columns):
        temp += 1 << (i * (rows + 1))
    return temp
BOTTOM = bottom_mask()

def full_mask():
    ## 全部棋盘位置mask
    temp = 0
    for i in range(columns):
        temp += ((1 << rows) - 1) << (i * (rows + 1))
    return temp
FULL = full_mask()

def get_bottom_mask(col):
    ## col列的底部值为1
    return 1 << (col * (rows + 1))

def get_column_mask(col):
    ## col列棋盘的全部mask
    return ((1 << rows) - 1) << (col * (rows + 1))

def get_top_mask(col):
    ## col列棋盘的顶部值为1
    return 1 << (rows - 1) << col * (rows + 1)


## 用于对棋盘整体移动
def UP(pos, i):
    return pos << i

def DOWN(pos, i):
    return pos >> i

def LEFT(pos, i):
    return pos >> (i * (rows + 1))

def RIGHT(pos, i):
    return pos << (i * (rows + 1))

def UP_LEFT(pos, i):
    return pos >> (i * rows)

def DOWN_RIGHT(pos, i):
    return pos << (i * rows)

def UP_RIGHT(pos, i):
    return pos << (i * (rows + 2))

def DOWN_LEFT(pos, i):
    return pos >> (i * (rows + 2))


def get_winning_moves(pos, mask):
    ## 计算可以获胜的位置
    temp = UP(pos, 1) & UP(pos, 2) & UP(pos, 3)
    temp |= LEFT(pos, 1) & LEFT(pos, 2) & LEFT(pos, 3)
    temp |= RIGHT(pos, 1) & LEFT(pos, 1) & LEFT(pos, 2)
    temp |= RIGHT(pos, 2) & RIGHT(pos, 1) & LEFT(pos, 1)
    temp |= RIGHT(pos, 3) & RIGHT(pos, 2) & RIGHT(pos, 1)
    temp |= UP_LEFT(pos, 1) & UP_LEFT(pos, 2) & UP_LEFT(pos, 3)
    temp |= DOWN_RIGHT(pos, 1) & UP_LEFT(pos, 1) & UP_LEFT(pos, 2)
    temp |= DOWN_RIGHT(pos, 2) & DOWN_RIGHT(pos, 1) & UP_LEFT(pos, 1)
    temp |= DOWN_RIGHT(pos, 3) & DOWN_RIGHT(pos, 2) & DOWN_RIGHT(pos, 1)
    temp |= UP_RIGHT(pos, 1) & UP_RIGHT(pos, 2) & UP_RIGHT(pos, 3)
    temp |= DOWN_LEFT(pos, 1) & UP_RIGHT(pos, 1) & UP_RIGHT(pos, 2)
    temp |= DOWN_LEFT(pos, 2) & DOWN_LEFT(pos, 1) & UP_RIGHT(pos, 1)
    temp |= DOWN_LEFT(pos, 3) & DOWN_LEFT(pos, 2) & DOWN_LEFT(pos, 1)
    return temp & (FULL ^ mask)

def count_winning_moves(pos, mask):
    ## 计算可以获胜的点的个数
    moves = get_winning_moves(pos, mask)
    n = 0
    while moves:
        moves &= moves - 1
        n += 1
    return n

def move_to_column(move):
    ## 二进制move变成col
    action = []
    for i in range(columns):
        mask = get_column_mask(i)
        if mask & move:
            action.append(i)
    return action


class State:
    def __init__(self, observation, pos = None, mask = None, mark = None):
        # pos,mask为棋盘二进制标记，mark为当前下的人，initmark为初始时刻下的人
        if mark is not None:
            self.mark = mark
        else:
            self.mark = observation.mark
        if pos is not None and mask is not None:
            self.pos = pos
            self.mask = mask
        else:
            self.pos, self.mask = self.board_to_pos(observation.board, self.mark)
        self.initmark = observation.mark
        self.observation = observation
        self.board_score = count_winning_moves(self.pos ^ self.mask, self.mask) * 0.15
        self.legal_moves = self.get_legal_moves()

    def board_to_pos(self, board, mark):
        ## board变成二进制棋盘记录pos,mask，mark为当前下的人
        pos = 0
        mask = 0
        for i in range(rows):
            for j in range(columns):
                if board[i * columns + j] == mark:
                    pos += 1 << (j * (rows + 1) + (rows - 1 - i))
                if board[i * columns + j] != 0:
                    mask += 1 << (j * (rows + 1) + (rows - 1 - i))
        return pos, mask
    
    def terminal(self):
        ## 判断是否终局
        if self.legal_moves == 0:
            return True
        if self.win():
            return True
        return False
    
    def win(self):
        ## 判断对手是否赢棋
        # 水平
        pos = self.pos ^ self.mask
        m = pos & (LEFT(pos, 1))
        if m & (LEFT(m, 2)):
            return True

        # 垂直
        m = pos & (DOWN(pos, 1))
        if m & (DOWN(m, 2)):
            return True

        # 对角线1
        m = pos & (UP_LEFT(pos, 1))
        if m & (UP_LEFT(m, 2)):
            return True

        # 对角线2 
        m = pos & (DOWN_LEFT(pos, 1))
        if m & (DOWN_LEFT(m, 2)):
            return True

        return False
       
    def get_possible_actions(self):
        ## 所有可以放置的列
        return move_to_column(self.legal_moves)

    def make_move(self, col):
        ## 在col动一步
        pos = self.pos ^ self.mask
        mask = self.mask | (self.mask + get_bottom_mask(col))
        return State(observation=self.observation, pos=pos, mask=mask, mark=3-self.mark)

    def get_legal_moves(self):
        ## 二进制的可移动位置
        return (self.mask + BOTTOM) & FULL
    
    def get_forced_moves(self):
        ## 必须堵住的位置
        opponent_winning_moves = get_winning_moves(self.pos ^ self.mask, self.mask) ## 对手可以赢的位置
        legal_moves = self.legal_moves ## 可下的位置
        forced_moves = legal_moves & opponent_winning_moves
        return forced_moves
    
    def get_non_losing_moves(self):
        ## 两步之内不会输的方法 
        opponent_winning_moves = get_winning_moves(self.pos ^ self.mask, self.mask) ## 对手可以赢的位置
        legal_moves = self.legal_moves  ## 可下的位置
        forced_moves = legal_moves & opponent_winning_moves
        if forced_moves:
            if forced_moves & (forced_moves - 1):
                ## 有多于一个必须下的位置
                return move_to_column(0) ## 准备返回空列表
            legal_moves = forced_moves

        return move_to_column(legal_moves & ~(opponent_winning_moves >> 1)) ## 必须下的位置，并且自己的一步棋不会让对手胜利
    
    def can_win_with_onemove(self):
        ## 是否可以一步胜利
        self.winning_moves = get_winning_moves(self.pos, self.mask)
        return self.winning_moves & self.legal_moves
    
    def get_reward(self):
        ## 计算得分
        if self.win():
            if self.initmark == self.mark:
                return 0
            else:
                return 2
        else:
            return 1
        
    def get_current_player(self):
        ## 是否为初始玩家
        if self.mark == self.initmark:
            return 1
        else:
            return -1


class TreeNode:
    def __init__(self, state, parent=None):
        self.state = state ## 棋盘
        self.isTerminal = state.terminal() ## 是否终局
        self.isFullyExpanded = self.isTerminal ## 是否展开完毕
        self.parent = parent ## 父节点
        self.num_visits = 0 ## 访问次数
        self.total_reward = 0 ## 总reward
        self.children = {} ## 子节点
    

def OurPolicy(state):
    ## 模拟策略
    while not state.terminal():
        ## 有必胜选必胜
        if state.can_win_with_onemove():
            if state.initmark == state.mark:
                return 2
            else:
                return 0
        ## 没必胜选不败
        action_list = state.get_non_losing_moves()
        if len(action_list) == 0:
            ## 没不败代表输了
            if state.initmark == state.mark:
                return 0
            else:
                return 2
        ## 随机模拟
        weights = [1 for x in action_list]
        action = choices(action_list, weights)[0]
        state = state.make_move(action)
    return state.get_reward()


class MCTS:
    def __init__(self, time_limit=None, total_time=None, explorationConstant=1 / np.sqrt(2),
                 simulatePolicy=OurPolicy):
        self.time_limit = time_limit
        self.total_time = total_time
        self.explorationConstant = explorationConstant ## UCB常数
        self.simulate = simulatePolicy ## 模拟策略

    def search(self, initialState):
        ## MCTS搜索
        self.root = TreeNode(initialState, None)
        if self.total_time >= 30:
            time_limit = time.time() + self.time_limit + 4
        elif self.total_time >= 3.5:
            time_limit = time.time() + self.time_limit + 3
        else:
            time_limit = time.time() + self.time_limit

        while time.time() < time_limit - 0.002:
            self.one_round()

        best_child = self.get_best_child(self.root, 0)
        action= (action for action, node in self.root.children.items() if node is best_child).__next__()
        return action

    def one_round(self):
        ## 进行一次探索
        node = self.select_node(self.root)
        reward = self.simulate(node.state)
        self.backpropogate(node, reward)

    def select_node(self, node):
        ## 选择要模拟的节点
        while not node.isTerminal:
            if node.isFullyExpanded:
                ## 当前节点被探索完毕，选最好的子节点
                node = self.get_best_child(node, self.explorationConstant)
            else:
                ## 遇到未探索完毕的节点，扩展一个子节点
                return self.expand(node)
        return node

    def expand(self, node):
        ## 扩展
        if node.state.can_win_with_onemove():
            ## 有必胜选必胜
            actions = [move_to_column(node.state.can_win_with_onemove())[0]]
        else:
            ## 没必胜选不输
            actions = node.state.get_non_losing_moves()
            if len(actions) == 0:
                ## 没不输随便走
                actions = [node.state.get_possible_actions()[0]]
        for action in actions:
            if action not in node.children:
                ## 在可选择的行动中扩展一个子节点
                new_node = TreeNode(node.state.make_move(action), node)
                node.children[action] = new_node
                new_node.parent = node
                ## 如果这是最后一个未扩展的子节点，更新父节点状态为全部扩展
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return new_node

    def backpropogate(self, node, reward):
        ## 回溯
        while node is not None:
            node.num_visits += 1
            node.total_reward += reward
            node = node.parent

    def get_best_child(self, node, explorationValue):
        ## 选择最好的子节点
        best_value = float("-inf")
        best_child = []
        for child in node.children.values():
            ## 按照UCB最大进行选择
            value = node.state.get_current_player() * child.total_reward / child.num_visits + child.state.board_score + explorationValue * np.sqrt(
                2 * np.log(node.num_visits) / child.num_visits)
            if value > best_value:
                best_value = value
                best_child = [child]
            ## 多个UCB值相同的节点随机选一个
            elif value == best_value:
                best_child.append(child)
        return choice(best_child)


def my_agent(observation, configuration):
    global columns
    global rows
    global BOTTOM
    global FULL
    columns = configuration.columns
    rows = configuration.rows
    BOTTOM = bottom_mask()
    FULL = full_mask()

    state = State(observation, configuration)
    ## 有必胜步走必胜步
    if state.can_win_with_onemove():
        return move_to_column(state.can_win_with_onemove())[0]
    ## 对手有必胜步要阻止
    if state.get_forced_moves():
        return move_to_column(state.get_forced_moves())[0]
    ## 都没有进行MCTS
    m = MCTS(time_limit=configuration.actTimeout, total_time=observation.remainingOverageTime, simulatePolicy=OurPolicy)
    action = m.search(initialState=state)
    return action
