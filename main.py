import kandinsky as k
import ion as i
import time as t
import random as r

def clear_screen() -> None:
    k.fill_rect(0, 0, 320, 240, Color.WHITE)

def idx_to_pos(x_y: list[int, int]) -> tuple[int, int]:
    return (x_y[0]*25+60, x_y[1]*25+11)

class Color:
    WHITE = k.color(255, 255, 255)
    BLACK = k.color(0, 0, 0)

    BACKGROUND = k.color(99, 49, 0)
    SEPARATION = k.color(251, 245, 233)
    SELECT = k.color(0, 255, 0)

    BUTTONLINE = k.color(80, 80, 80)
    BUTTONCHOOSEN = k.color(168, 168, 168)

    class board:
        MOVE_TO = k.color(255, 0, 0)
        P_MOVES = k.color(61, 61, 61)
        WHITE_P = k.color(244, 227, 195)
        WHITE = k.color(177, 126, 32)
        BLACK = k.color(99, 71, 18)
        BLACK_P = k.color(0, 0, 0)

class infoMap:
    pawn: tuple[int, int, int] = (
        (9, 11, 3), (10, 10, 5), (11, 9, 7), (12, 9, 7), (13, 9, 7), 
        (14, 10, 5), (15, 10, 5), (16, 11, 3), (17, 10, 5), (18, 9, 7), 
        (19, 8, 9), (20, 8, 9), (21, 7, 11), (22, 6, 13), (23, 6, 13)
    )
    rook: tuple[int, int, int] = (
        (2, 4, 3), (2, 9, 3), (2, 14, 3), (2, 19, 3), (3, 4, 3), 
        (3, 9, 3), (3, 14, 3), (3, 19, 3), (4, 4, 18), (5, 4, 18), 
        (6, 4, 18), (7, 4, 18), (8, 5, 16), (9, 6, 14), (10, 8, 10), 
        (11, 8, 10), (12, 8, 10), (13, 8, 10), (14, 8, 10), (15, 7, 12), 
        (16, 6, 14), (17, 5, 16), (18, 5, 16), (19, 5, 16), (20, 5, 16), 
        (21, 4, 18), (22, 4, 18), (23, 4, 18)
    )
    knight: tuple[int, int, int] = (
        (1, 13, 3), (2, 10, 7), (3, 9, 9), (4, 8, 11), (5, 7, 13), 
        (6, 6, 14), (7, 5, 16), (8, 4, 17), (9, 3, 18), (10, 2, 10), 
        (10, 14, 8), (11, 2, 8), (11, 14, 8), (12, 3, 5), (12, 14, 8), 
        (13, 13, 9), (14, 12, 10), (15, 11, 12), (16, 10, 13), (17, 9, 14), 
        (18, 8, 15), (19, 7, 16), (20, 7, 16), (21, 6, 17), (22, 6, 18), 
        (23, 5, 19)
    )
    bishop: tuple[int, int, int] = (
        (1, 12, 1), (2, 11, 3), (3, 12, 1), (4, 11, 3), (5, 10, 5), 
        (6, 8, 9), (7, 6, 13), (8, 5, 7), (8, 13, 7), (9, 4, 8), 
        (9, 13, 8), (10, 4, 8), (10, 13, 8), (11, 4, 5), (11, 16, 5), 
        (12, 4, 8), (12, 13, 8), (13, 4, 8), (13, 13, 8), (14, 5, 7), 
        (14, 13, 7), (15, 6, 13), (16, 7, 10), (17, 8, 8), (18, 8, 8), 
        (19, 7, 10), (20, 5, 13), (21, 4, 8), (21, 13, 7), (22, 2, 7), 
        (22, 16, 6), (23, 1, 5), (23, 19, 4)
    )
    queen: tuple[int, int, int] = (
        (1, 12, 1), (2, 4, 1), (2, 11, 3), (2, 20, 1), (3, 3, 3), 
        (3, 12, 1), (3, 19, 3), (4, 4, 1), (4, 11, 3), (4, 20, 1), 
        (5, 3, 3), (5, 10, 5), (5, 19, 3), (6, 3, 4), (6, 10, 5), 
        (6, 18, 4), (7, 2, 5), (7, 9, 7), (7, 18, 5), (8, 2, 5), 
        (8, 9, 7), (8, 18, 5), (9, 2, 6), (9, 9, 7), (9, 17, 6), 
        (10, 2, 6), (10, 9, 7), (10, 17, 6), (11, 2, 21), (12, 2, 21), 
        (13, 3, 19), (14, 3, 19), (15, 3, 19), (16, 4, 17), (17, 5, 15), 
        (18, 5, 15), (19, 4, 17), (20, 3, 19), (21, 3, 19), (22, 3, 19), 
        (23, 3, 19)
    )
    king: tuple[int, int, int] = (
        (1, 12, 1), (2, 11, 3), (3, 12, 1), (4, 5, 3), (4, 11, 3), 
        (4, 17, 3), (5, 4, 5), (5, 10, 5), (5, 16, 5), (6, 3, 6), 
        (6, 10, 5), (6, 16, 6), (7, 2, 8), (7, 11, 3), (7, 15, 8), 
        (8, 2, 9), (8, 12, 1), (8, 14, 9), (9, 2, 10), (9, 13, 10), 
        (10, 2, 10), (10, 13, 10), (11, 2, 10), (11, 13, 10), (12, 2, 10), 
        (12, 13, 10), (13, 3, 9), (13, 13, 9), (14, 7, 5), (14, 13, 5), 
        (15, 5, 2), (15, 11, 1), (15, 13, 1), (15, 18, 2), (16, 5, 6), 
        (16, 14, 6), (17, 5, 15), (18, 7, 11), (19, 4, 3), (19, 11, 3), 
        (19, 18, 3), (20, 3, 8), (20, 14, 8), (21, 3, 19), (22, 3, 19), 
        (23, 3, 19)
    )

    lookup_map: dict = {
        1: pawn,
        3: knight,
        4: bishop,
        5: rook,
        9: queen,
        1000: king
    }

class Board:
    def __init__(self):
        self.to_who: bool = True # white = True; else False

    def __bot_move(self, board: list[list[int]], depth: int, col, alpha, beta, n_moves: int, top=True):
        
        all_moves = self.get_all_moves(board, col)

        if depth == 0 or not all_moves: return self.__bot_score(board, n_moves, col)

        best_move = None

        best_value = float("-inf") if col == 1 else float("inf")
        for pos, moves in all_moves:
            for m in moves:
                new_board = self.__simulate_move(board, pos, m)

                value = self.__bot_move(new_board, depth-1, -1, alpha, beta, n_moves+1, False)

                if (value > best_value and col == 1) or (value < best_value and col == -1):
                    best_value = value
                    if top:
                        best_move = (pos, m)

                if col == 1: alpha = max(alpha, value)
                else:        beta  = min(beta, value)

                if beta <= alpha:
                    break

        return best_move if top else best_value

    def __bot_score(self, board: list[list[int]], moves: int, col: int) -> int:
        val = self.evaluate_board(board) * 2
        val += 10000 if self.is_checkmate(board) else -10000
        if moves < 12:
            val += len(self.get_all_moves(board, col)) * 2
        return val + 0.001 * r.random()

    def __simulate_move(self, board, frm, to):
        new_board = [row[:] for row in board]

        piece = new_board[frm[1]][frm[0]]
        new_board[frm[1]][frm[0]] = 0
        new_board[to[1]][to[0]] = piece
        return new_board

    def __get_all_moves_from(self, pos: tuple[int, int], piece: int, board: list[list[int]]) -> list[tuple[int, int]]:
        moves = []
        x, y = pos
        color = 1 if piece > 0 else -1
        abs_piece = abs(piece)

        def inside(nx, ny):
            return 0 <= nx < 8 and 0 <= ny < 8

        def add_slide(directions):
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                while inside(nx, ny):
                    if board[ny][nx] == 0:
                        moves.append((nx, ny))
                    elif board[ny][nx] * color < 0:
                        moves.append((nx, ny))
                        break
                    else:
                        break
                    nx += dx
                    ny += dy

        # === PION ===
        if abs_piece == 1:
            dir = -1 if color == 1 else 1  # blanc monte, noir descend
            start_row = 6 if color == 1 else 1

            if inside(x, y + dir) and board[y + dir][x] == 0:
                moves.append((x, y + dir))
                if y == start_row and board[y + 2 * dir][x] == 0:
                    moves.append((x, y + 2 * dir))

            for dx in (-1, 1):
                nx, ny = x + dx, y + dir
                if inside(nx, ny) and board[ny][nx] * color < 0:
                    moves.append((nx, ny))

        # === CAVALIER ===
        elif abs_piece == 3:
            for dx, dy in (
                (-2, -1), (-2, 1), (2, -1), (2, 1),
                (-1, -2), (-1, 2), (1, -2), (1, 2)
            ):
                nx, ny = x + dx, y + dy
                if inside(nx, ny) and board[ny][nx] * color <= 0:
                    moves.append((nx, ny))

        # === FOU ===
        elif abs_piece == 4:
            add_slide(((1, 1), (1, -1), (-1, 1), (-1, -1)))

        # === TOUR ===
        elif abs_piece == 5:
            add_slide(((1, 0), (-1, 0), (0, 1), (0, -1)))

        # === DAME ===
        elif abs_piece == 9:
            add_slide((
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)
            ))

        # === ROI ===
        elif abs_piece == 1000:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if inside(nx, ny) and board[ny][nx] * color <= 0:
                        moves.append((nx, ny))

        del x, y, color, abs_piece

        return moves

    def find_king(self, board: list[list[int]]) -> tuple[int, int]:
        for y in range(0, 8, 1):
            for x in range(0, 8, 1):
                if board[y][x] == (1000 if self.to_who else -1000):
                    return (x, y)
        
        del x, y

    def is_square_attacked(self, board: list[list[int]], pos: tuple[int, int], col: int) -> bool:
        for x in range(0, 8, 1):
            for y in range(0, 8, 1):
                p = board[y][x]
                if p == 0: continue 
                if (1 if p > 0 else -1) != col: continue
                moves = self.__get_all_moves_from((x, y), p, board)
                for mx, my in moves:
                    if mx == pos[0] and my == pos[1]:
                        del mx, my, moves, p, x, y
                        return True
        del p, x, y
        return False

    def is_check(self, board: list[list[int]]) -> bool:
        k_pos = self.find_king(board)
        if not k_pos: return True
        return self.is_square_attacked(board, k_pos, -1 if self.to_who else 1)

    def is_checkmate(self, board: list[list[int]]) -> bool:
        if not self.is_check(board): return False
        return len(self.get_all_moves(board, 1 if self.to_who else -1)) == 0
    
    def is_stalemate(self, board: list[list[int]]) -> bool:
        if self.is_check(board): return False
        return len(self.get_all_moves(board, 1 if self.to_who else -1)) == 0
    
    def is_only_kings(self, board: list[list[int]]) -> bool:
        return all([abs(board[y][x]) == 1000 for y in range(0, 8, 1) for x in range(0, 8, 1) if board[y][x] != 0])

    def create_board(self) -> list[list[int]]:
        return [[-5, -3, -4, -9, -1000, -4, -3, -5],
                [-1, -1, -1, -1,    -1, -1, -1, -1],
                [0,   0,  0,  0,     0,  0,  0,  0],
                [0,   0,  0,  0,     0,  0,  0,  0],
                [0,   0,  0,  0,     0,  0,  0,  0],
                [0,   0,  0,  0,     0,  0,  0,  0],
                [1,   1,  1,  1,     1,  1,  1,  1],
                [5,   3,  4,  9,  1000,  4,  3,  5]]

    def evaluate_board(self, board: list[list[int]]) -> int:
        return sum([sum(line) for line in board])

    def get_poses_from(self, board: list[list[int]], pos: list[int]) -> list[tuple[int, int]]:
        piece: int = board[pos[1]][pos[0]]
        if piece == 0: return []
        moves = self.__get_all_moves_from(pos, piece, board)
        legals: list[tuple[int, int]] = []
        for (mx, my) in moves:
            move_board = [row[:] for row in board]
            self.move(pos, [mx, my], move_board)
            self.to_who = not self.to_who
            if not self.is_check(move_board):
                legals.append((mx, my))

        del piece
        return legals

    def get_all_moves(self, board: list[list[int]], col: int) -> list[tuple[int, int]]:
        all_moves = []
        for x in range(0, 8, 1):
            for y in range(0, 8, 1):
                piece = board[y][x]
                if piece == 0: continue
                if (1 if piece > 0 else -1) != col: continue
                all_moves.append(((x, y), self.get_poses_from(board, (x, y))))
        return all_moves

    def make_move(self, board: list[list[int]], n_moves: int):
        return self.__bot_move(board, depth=2, col=-1, alpha=float("-inf"), beta=float("inf"), n_moves=n_moves)

    def move(self, piece: list[int, int], to: list[int, int], board: list[list[int]]):
        p = board[piece[1]][piece[0]]
        board[piece[1]][piece[0]] = 0
        board[to[1]][to[0]] = p
        self.to_who = not self.to_who
        return p
    
    def is_legal(self, piece: int) -> bool:
        return (self.to_who and piece > 0) or (not self.to_who and piece < 0)

class Menu:
    
    def __init__(self):
        self.IsMenu: bool = True
        self.select: bool = False

    def show_menu(self):
        k.fill_rect(0, 0, 320, 240, Color.BACKGROUND)
        k.draw_string("NUMWORKS CHESS", 90, 20, Color.WHITE, Color.BACKGROUND)
        self.refresh_menu()

    def refresh_menu(self):
        k.fill_rect(85, 60, 150, 60, Color.BUTTONLINE)
        k.fill_rect(91, 66, 138, 48, Color.WHITE if self.select else Color.BUTTONCHOOSEN)
        k.draw_string("1 VS 1", 130, 82, Color.BLACK , Color.WHITE if self.select else Color.BUTTONCHOOSEN)
        k.fill_rect(85, 140, 150, 60, Color.BUTTONLINE)
        k.fill_rect(91, 146, 138, 48, Color.WHITE if not self.select else Color.BUTTONCHOOSEN)
        k.draw_string("1 VS BOT", 120, 162, Color.BLACK, Color.WHITE if not self.select else Color.BUTTONCHOOSEN)

class Game:

    def __init__(self):
        self.IsGame: bool = True

        self.board: Board = Board()
        self.plate: list[list[int]] = self.board.create_board()

        self.sel_idx: list[int, int] = [4, 7]
        self.old_sel_idx: list[int, int] = [4, 7]
        self.sel_col: tuple[int, int, int] = Color.board.BLACK
        self.sel_poses: list[tuple[int, int]] = []
        self.piece_sel: list[int] = []
        self.sel_mode: bool = False

        self.movements: int = 0
        self.is_bot: bool = False

    def __show_result(self):
        k.fill_rect(110, 61, 100, 100, Color.BLACK)
        k.fill_rect(112, 63, 96, 96, Color.WHITE)
        k.draw_string(str(self.movements), int(160 - ((len(str(self.movements))*10-1)/2)), 103, Color.BLACK)
        k.draw_string("MOVES", 135, 120, Color.BLACK)

    def __show_checkmate(self):
        self.__show_result()
        k.draw_string("BLACK" if self.board.to_who else "WHITE" + " WON", 115, 63, Color.BLACK)
        k.draw_string("CHECKMATE", 115, 80, Color.BLACK)
        
    def __show_stalemate(self):
        self.__show_result()
        k.draw_string("NONE WON", 120, 63, Color.BLACK)
        k.draw_string("STALEMATE", 115, 80, Color.BLACK)

    def __show_draw(self):
        self.__show_result()
        k.draw_string("NONE WON", 120, 63, Color.BLACK)
        k.draw_string("DRAW", 140, 80, Color.BLACK)

    def __draw_piece(self, piece: int, pos: tuple[int, int]):
        p = infoMap.lookup_map.get(abs(piece), None)
        if not p: return
        for info in p:
            k.fill_rect(pos[0]+info[1], pos[1]+info[0], info[2], 1, Color.board.BLACK_P if piece < 0 else Color.board.WHITE_P)

        del p, info

    def __draw_pieces(self):

        for x in range(0, 8, 1):
            for y in range(0, 8, 1):
                self.__draw_piece(self.plate[y][x], idx_to_pos([x, y]))

        del x, y
    
    def  __refresh_move(self, fm, to):
        piece = idx_to_pos(fm)
        k.fill_rect(piece[0], piece[1], 25, 25, k.get_pixel(piece[0]+1, piece[1]+1))
        piece = idx_to_pos(to)
        k.fill_rect(piece[0], piece[1], 25, 25, k.get_pixel(piece[0]+1, piece[1]+1))

    def __draw_board(self):
        IsWhite: bool = True

        for y in range(11, 211, 25):
            for x in range(60, 260, 25):
                k.fill_rect(x, y, 25, 25, Color.board.WHITE if IsWhite else Color.board.BLACK)
                IsWhite = not IsWhite
            IsWhite = not IsWhite

        del IsWhite, x, y

    def __draw_around_square(self, pos, color):
        k.fill_rect(pos[0], pos[1], 25, 1, color)
        k.fill_rect(pos[0], pos[1], 1, 25, color)
        k.fill_rect(pos[0]+24, pos[1], 1, 25, color)
        k.fill_rect(pos[0], pos[1]+24, 25, 1, color)

    def set_mode(self, mode: bool):
        self.is_bot = mode

    def MakeBoard(self):
        k.fill_rect(0, 0, 320, 240, Color.BACKGROUND)
        k.fill_rect(55, 0, 210, 222, Color.SEPARATION)

        self.__draw_board()
        self.__draw_pieces()

    def show_select(self):
        self.__draw_around_square(idx_to_pos(self.sel_idx), Color.SELECT)

    def select_refresh(self):
        if tuple(self.old_sel_idx) in self.sel_poses:
            self.__draw_around_square(idx_to_pos(self.old_sel_idx), Color.board.MOVE_TO)
        else:
            self.__draw_around_square(idx_to_pos(self.old_sel_idx), self.sel_col)
        real_pos = idx_to_pos(self.sel_idx)
        self.__draw_around_square(real_pos, Color.SELECT)

        self.sel_col = k.get_pixel(real_pos[0]+1, real_pos[1]+1)
        self.old_sel_idx = self.sel_idx.copy()

        del real_pos

    def action_from_select(self):
        if not self.sel_mode:
            if not self.board.is_legal(self.plate[self.sel_idx[1]][self.sel_idx[0]]): return
            self.sel_poses = self.board.get_poses_from(self.plate, self.sel_idx)
            for pos in self.sel_poses:
                self.__draw_around_square(idx_to_pos(pos), Color.board.MOVE_TO)
            self.sel_mode = True
            self.piece_sel = self.sel_idx.copy()
            del pos
            return

        if tuple(self.sel_idx) not in self.sel_poses and self.sel_idx != self.piece_sel:
            for pos in self.sel_poses:
                og_col = idx_to_pos(pos)
                self.__draw_around_square(idx_to_pos(pos), k.get_pixel(og_col[0]+1, og_col[1]+1))
            self.sel_mode = False
            del og_col
            return

        if self.sel_idx == self.piece_sel: return
        p_val = self.plate[self.piece_sel[1]][self.piece_sel[0]]
        if not self.board.is_legal(p_val): return

        self.board.move(self.piece_sel, self.sel_idx, self.plate)
        self.__refresh_move(self.piece_sel, self.sel_idx)
        self.__draw_piece(p_val, idx_to_pos(self.sel_idx))

        for pos in self.sel_poses:
            if pos == tuple(self.sel_idx): continue
            og_col = idx_to_pos(pos)
            self.__draw_around_square(idx_to_pos(pos), k.get_pixel(og_col[0]+1, og_col[1]+1))

        if self.is_bot:
            self.verify_state()
            move = self.board.make_move(self.plate, self.movements)
            if not move: return
            p_val = self.plate[move[0][1]][move[0][0]]
            self.board.move(move[0], move[1], self.plate)
            piece = idx_to_pos(move[0])
            k.fill_rect(piece[0], piece[1], 25, 25, k.get_pixel(piece[0]+1, piece[1]+1))
            piece = idx_to_pos(move[1])
            k.fill_rect(piece[0], piece[1], 25, 25, k.get_pixel(piece[0]+1, piece[1]+1))
            self.__draw_piece(p_val, idx_to_pos(move[1]))

        self.select_refresh()
        self.sel_poses = []
        self.sel_mode = False
        self.movements += 1

        del p_val, piece
    
    def verify_state(self):

        score = self.board.evaluate_board(self.plate)
        k.fill_rect(265, 0, 55, 222, Color.BACKGROUND)
        k.fill_rect(0, 0, 55, 222, Color.BACKGROUND)
        if score > 0:
            k.draw_string(str(score), int(295 - ((len(str(score))*10-1)/2)), 111, Color.BLACK)
        if score < 0:
            k.draw_string(str(abs(score)), int(25 - ((len(str(abs(score)))*10-1)/2)), 111, Color.BLACK)

        if self.board.is_checkmate(self.plate):
            self.IsGame = False
            self.__show_checkmate()
            return
        if self.board.is_stalemate(self.plate):
            self.IsGame = False
            self.__show_stalemate()
            return
        if self.board.is_only_kings(self.plate):
            self.IsGame = False
            self.__show_draw()
            return
        
        del score


menu = Menu()
menu.show_menu()

while menu.IsMenu:
    if i.keydown(i.KEY_DOWN) or i.keydown(i.KEY_UP):
        menu.select = not menu.select
        menu.refresh_menu()
        t.sleep(0.2)
    if i.keydown(i.KEY_OK) or i.keydown(i.KEY_EXE):
        menu.IsMenu = False


game = Game()

game.MakeBoard()
game.show_select()
game.set_mode(menu.select)

refresh_point = t.monotonic()
move_wanted =  0

del menu
game.verify_state()


t.sleep(0.5)
while game.IsGame:

    if refresh_point + 0.2 <= t.monotonic():
        for key, val in ((i.KEY_LEFT,1), (i.KEY_RIGHT,2), (i.KEY_UP,3), (i.KEY_DOWN,4), (i.KEY_OK,5), (i.KEY_EXE,5)):
            if i.keydown(key):
                move_wanted = val
                break

        if move_wanted != 0:
            if move_wanted == 1 and game.sel_idx[0] != 0:
                game.sel_idx[0] -= 1
            elif move_wanted == 2 and game.sel_idx[0] != 7:
                game.sel_idx[0] += 1
            elif move_wanted == 3 and game.sel_idx[1] != 0:
                game.sel_idx[1] -= 1
            elif move_wanted == 4 and game.sel_idx[1] != 7:
                game.sel_idx[1] += 1
            elif move_wanted == 5 or move_wanted == 6:
                print("okay")
                game.action_from_select()
                game.verify_state()

            if move_wanted < 5:
                game.select_refresh()
            refresh_point = t.monotonic()
            move_wanted = 0

while True:
    if i.keydown(i.KEY_OK) or i.keydown(i.KEY_EXE):
        break