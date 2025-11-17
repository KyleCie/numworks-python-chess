import kandinsky as k
import ion as i
import time as t
from random import random

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
        (9, 11, 3), (10, 10, 5), (11, 9, 7, 3), (14, 10, 5, 2), (16, 11, 3), 
        (17, 10, 5), (18, 9, 7), (19, 8, 9, 2), (21, 7, 11), (22, 6, 13, 2) 
    )
    rook: tuple[int, int, int] = (
        (2, 4, 3, 2), (2, 9, 3, 2), (2, 14, 3, 2), (2, 19, 3), (3, 19, 3), 
        (4, 4, 18, 4), (8, 5, 16), (9, 6, 14), (10, 8, 10, 5), (15, 7, 12),
        (16, 6, 14), (17, 5, 16, 4), (21, 4, 18, 3)
    )
    knight: tuple[int, int, int] = (
        (1, 13, 3), (2, 10, 7), (3, 9, 9), (4, 8, 11), (5, 7, 13), 
        (6, 6, 14), (7, 5, 16), (8, 4, 17), (9, 3, 18), (10, 2, 10), 
        (10, 14, 8, 3), (11, 2, 8), (12, 3, 5), (13, 13, 9), (14, 12, 10), 
        (15, 11, 12), (16, 10, 13), (17, 9, 14), (18, 8, 15), (19, 7, 16, 2), 
        (21, 6, 17), (22, 6, 18), (23, 5, 19)
    )
    bishop: tuple[int, int, int] = (
        (1, 12, 1), (2, 11, 3), (3, 12, 1), (4, 11, 3), (5, 10, 5), 
        (6, 8, 9), (7, 6, 13), (8, 5, 7), (8, 13, 7), (9, 4, 8, 2), 
        (9, 13, 8, 2), (11, 4, 5), (11, 16, 5), (12, 4, 8), (12, 13, 8), 
        (13, 4, 8), (13, 13, 8), (14, 5, 7), (14, 13, 7), (15, 6, 13), 
        (16, 7, 10), (17, 8, 8, 2), (19, 7, 10), (20, 5, 13), (21, 4, 8), 
        (21, 13, 7), (22, 2, 7), (22, 16, 6), (23, 1, 5), (23, 19, 4)
    )
    queen: tuple[int, int, int] = (
        (1, 12, 1), (2, 4, 1), (2, 11, 3), (2, 20, 1), (3, 3, 3), 
        (3, 12, 1), (3, 19, 3), (4, 4, 1), (4, 11, 3), (4, 20, 1), 
        (5, 3, 3), (5, 10, 5, 2), (5, 19, 3), (6, 3, 4), (6, 18, 4), 
        (7, 2, 5, 2), (7, 9, 7, 4), (7, 18, 5, 2), (9, 2, 6, 2), (9, 17, 6, 2), 
        (11, 2, 21, 2), (13, 3, 19, 3), (16, 4, 17), (17, 5, 15), (18, 6, 14), 
        (19, 5, 16), (20, 3, 19, 4)
    )
    king: tuple[int, int, int] = (
        (1, 12, 1), (2, 11, 3), (3, 12, 1), (4, 5, 3), (4, 11, 3), 
        (4, 17, 3), (5, 4, 5), (5, 10, 5, 2), (5, 16, 5), (6, 3, 6), 
        (6, 16, 6), (7, 2, 8), (7, 11, 3), (7, 15, 8), (8, 2, 9), 
        (8, 12, 1), (8, 14, 9), (9, 2, 10, 4), (9, 13, 10, 4), (13, 3, 9), 
        (13, 13, 9), (14, 7, 5), (14, 13, 5), (15, 5, 2), (15, 11, 1), 
        (15, 13, 1), (15, 18, 2), (16, 5, 6), (16, 14, 6), (17, 5, 15), 
        (18, 7, 11), (19, 4, 3), (19, 11, 3), (19, 18, 3), (20, 3, 8), 
        (20, 14, 8), (21, 3, 19, 3)
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
        all_moves = self.get_all_moves(board,col)
        if depth == 0 or not all_moves: return self.__bot_score(board,n_moves,col)
        best_move = None
        best_val = float("-inf") if col == 1 else float("inf")
        for pos,moves in all_moves:
            for m in moves:
                b = self.__simulate_move(board,pos,m)
                v = self.__bot_move(b,depth-1,-col,alpha,beta,n_moves+1,False)
                if (col == 1 and v>best_val) or (col == -1 and v<best_val):
                    best_val = v
                    if top: best_move=(pos,m)
                if col == 1: alpha=max(alpha,v)
                else:        beta=min(beta,v)
                if beta<=alpha: break
        return best_move if top else best_val

    def __bot_score(self, board: list[list[int]], moves: int, col: int) -> int:
        val = self.evaluate_board(board) * 2
        val += 10000 if self.is_checkmate(board) else -10000
        if moves < 12: val += len(self.get_all_moves(board, col)) * 2
        return val + 0.001 * random()

    def __simulate_move(self, board, frm, to):
        new_board = [row[:] for row in board]
        piece = new_board[frm[1]][frm[0]]
        new_board[frm[1]][frm[0]] = 0
        new_board[to[1]][to[0]] = piece
        return new_board

    def __get_all_moves_from(self, pos: tuple[int, int], p: int, bd: list[list[int]]) -> list[tuple[int, int]]:
        m = []; x,y = pos; c=1 if p>0 else-1;a=abs(p)
        def ins(nx,ny): return 0<=nx<8 and 0<=ny<8
        def slide(ds):
            for dx,dy in ds:
                nx,ny=x+dx,y+dy
                while ins(nx,ny):
                    t=bd[ny][nx]
                    if t==0: m.append((nx,ny))
                    elif t*c<0: m.append((nx,ny)); break
                    else: break
                    nx+=dx;ny+=dy
        if a==1:
            d=-1 if c==1 else 1;sr=6 if c==1 else 1
            if ins(x,y+d)and bd[y+d][x]==0:
                m.append((x,y+d))
                if y==sr and bd[y+2*d][x]==0: m.append((x,y+2*d))
            for dx in(-1,1):
                nx,ny=x+dx,y+d
                if ins(nx,ny)and bd[ny][nx]*c<0: m.append((nx,ny))
        elif a==3:
            for dx,dy in((-2,-1),(-2,1),(2,-1),(2,1),(-1,-2),(-1,2),(1,-2),(1,2)):
                nx,ny=x+dx,y+dy
                if ins(nx,ny)and bd[ny][nx]*c<=0: m.append((nx,ny))
        elif a==4: slide(((1,1),(1,-1),(-1,1),(-1,-1)))
        elif a==5: slide(((1,0),(-1,0),(0,1),(0,-1)))
        elif a==9: slide(((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)))
        elif a==1000:
            for dx in(-1,0,1):
                for dy in(-1,0,1):
                    if dx or dy:
                        nx,ny=x+dx,y+dy
                        if ins(nx,ny) and bd[ny][nx]*c<=0: m.append((nx,ny))
        return m

    def find_king(self, board: list[list[int]]) -> tuple[int, int]:
        return next((x,y)for y in range(8)for x in range(8)if board[y][x]==(1000 if self.to_who else -1000))

    def is_square_attacked(self, board: list[list[int]], pos: tuple[int, int], col: int) -> bool:
        for y in range(8):
            for x in range(8):
                p=board[y][x]
                if p and (p>0)-(p<0)==col:
                    for mx,my in self.__get_all_moves_from((x,y),p,board):
                        if (mx,my)==pos: return True
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
        p = board[pos[1]][pos[0]]
        if not p: return []
        l = []
        for mx,my in self.__get_all_moves_from(pos,p,board):
            b = [r[:] for r in board]
            self.move(pos,[mx,my],b)
            self.to_who = not self.to_who
            if not self.is_check(b): l.append((mx,my))
        return l

    def get_all_moves(self, board: list[list[int]], col: int) -> list[tuple[int, int]]:
        all_moves = []
        for y in range(8):
            for x in range(8):
                p = board[y][x]
                if p and (p>0)-(p<0) == col: all_moves.append(((x,y),self.get_poses_from(board,(x,y))))
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
        for info in infoMap.lookup_map.get(abs(piece), ()):
            k.fill_rect(pos[0]+info[1], pos[1]+info[0], info[2], 1 if len(info) == 3 else info[3], Color.board.BLACK_P if piece < 0 else Color.board.WHITE_P)

    def __draw_pieces(self):
        for x in range(8):
            for y in range(8):
                self.__draw_piece(self.plate[y][x], idx_to_pos([x, y]))
    
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

    def action_from_select(self):
        if not self.sel_mode:
            if not self.board.is_legal(self.plate[self.sel_idx[1]][self.sel_idx[0]]): return
            self.sel_poses = self.board.get_poses_from(self.plate, self.sel_idx)
            for pos in self.sel_poses:
                self.__draw_around_square(idx_to_pos(pos), Color.board.MOVE_TO)
            self.sel_mode = True
            self.piece_sel = self.sel_idx.copy()
            return

        if tuple(self.sel_idx) not in self.sel_poses and self.sel_idx != self.piece_sel:
            for pos in self.sel_poses:
                og_col = idx_to_pos(pos)
                self.__draw_around_square(idx_to_pos(pos), k.get_pixel(og_col[0]+1, og_col[1]+1))
            self.sel_mode = False
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
            for m in move:
                piece = idx_to_pos(m)
                k.fill_rect(piece[0], piece[1], 25, 25, k.get_pixel(piece[0]+1, piece[1]+1))
            self.__draw_piece(p_val, idx_to_pos(move[1]))

        self.select_refresh()
        self.sel_poses = []
        self.sel_mode = False
        self.movements += 1
    
    def verify_state(self):

        score = self.board.evaluate_board(self.plate)
        k.fill_rect(265, 0, 55, 222, Color.BACKGROUND)
        k.fill_rect(0, 0, 55, 222, Color.BACKGROUND)
        k.draw_string(str(abs(score)), int((25 if score < 0 else 295) - ((len(str(abs(score)))*10-1)/2)), 111, Color.BLACK)

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