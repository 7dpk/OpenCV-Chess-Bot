import time
import os
from cv2 import cv2
import chess
import chess.engine
import numpy as np
import pyautogui
import random
import keyboard
import PIL.Image
import chess.polyglot
import d3dshot
import subprocess
from datetime import datetime

dshot = d3dshot.create(capture_output="numpy")


def _get_all_sequences(seq, min_seq_len=7, err_px=5):
    """ Given sequence of increasing numbers, get all sequences with common
        spacing (within err_px) that contain at least min_seq_len values
    """
    # Sanity check that there are enough values to satisfy
    if len(seq) < min_seq_len:
        return []

    # For every value, take the next value and see how many times we can step
    # that falls on another value within err_px points
    seqs = []
    for i in range(len(seq)-1):
        for j in range(i+1, len(seq)):
            # Check that seq[i], seq[j] not already in previous sequences
            duplicate = False
            for prev_seq in seqs:
                for k in range(len(prev_seq)-1):
                    if seq[i] == prev_seq[k] and seq[j] == prev_seq[k+1]:
                        duplicate = True
            if duplicate:
                continue
            d = seq[j] - seq[i]

            # Ignore two points that are within error bounds of each other
            if d < err_px:
                continue

            s = [seq[i], seq[j]]
            n = s[-1] + d
            while np.abs((seq-n)).min() < err_px:
                n = seq[np.abs((seq-n)).argmin()]
                s.append(n)
                n = s[-1] + d

            if len(s) >= min_seq_len:
                s = np.array(s)
                seqs.append(s)
    return seqs


def _nonmax_suppress_1d(arr, winsize=5):
    """ Return 1d array with only peaks, use neighborhood window of winsize px
    """
    _arr = arr.copy()
    for i in range(_arr.size):
        if i == 0:
            left_neighborhood = 0
        else:
            left_neighborhood = arr[max(0, i-winsize):i]
        if i >= _arr.size-2:
            right_neighborhood = 0
        else:
            right_neighborhood = arr[i+1:min(arr.size-1, i+winsize)]

        if arr[i] < np.max(left_neighborhood) or arr[i] <= np.max(right_neighborhood):
            _arr[i] = 0
    return _arr


def detect_chessboard_corners(img_arr_gray, noise_threshold=8000):
    """ Load image grayscale as an numpy array
        Return None on failure to find a chessboard

        noise_threshold: Ratio of standard deviation of hough values along an axis
        versus the number of pixels, manually measured bad trigger images
        at < 5,000 and good  chessboards values at > 10,000
    """
    # Get gradients, split into positive and inverted negative components
    gx, gy = np.gradient(img_arr_gray)
    gx_pos = gx.copy()
    gx_pos[gx_pos < 0] = 0
    gx_neg = -gx.copy()
    gx_neg[gx_neg < 0] = 0

    gy_pos = gy.copy()
    gy_pos[gy_pos < 0] = 0
    gy_neg = -gy.copy()
    gy_neg[gy_neg < 0] = 0

    # 1-D ampltitude of hough transform of gradients about X & Y axes
    num_px = img_arr_gray.shape[0] * img_arr_gray.shape[1]
    hough_gx = gx_pos.sum(axis=1) * gx_neg.sum(axis=1)
    hough_gy = gy_pos.sum(axis=0) * gy_neg.sum(axis=0)

    # Check that gradient peak signal is strong enough by
    # comparing normalized standard deviation to threshold
    if min(hough_gx.std() / hough_gx.size,
           hough_gy.std() / hough_gy.size) < noise_threshold:
        return None

    # Normalize and skeletonize to just local peaks
    hough_gx = _nonmax_suppress_1d(hough_gx) / hough_gx.max()
    hough_gy = _nonmax_suppress_1d(hough_gy) / hough_gy.max()

    # Arbitrary threshold of 20% of max
    hough_gx[hough_gx < 0.2] = 0
    hough_gy[hough_gy < 0.2] = 0

    # Now we have a set of potential vertical and horizontal lines that
    # may contain some noisy readings, try different subsets of them with
    # consistent spacing until we get a set of 7, choose strongest set of 7
    pot_lines_x = np.where(hough_gx)[0]
    pot_lines_y = np.where(hough_gy)[0]
    pot_lines_x_vals = hough_gx[pot_lines_x]
    pot_lines_y_vals = hough_gy[pot_lines_y]

    # Get all possible length 7+ sequences
    seqs_x = _get_all_sequences(pot_lines_x)
    seqs_y = _get_all_sequences(pot_lines_y)

    if len(seqs_x) == 0 or len(seqs_y) == 0:
        return None

    # Score sequences by the strength of their hough peaks
    seqs_x_vals = [pot_lines_x_vals[[v in seq for v in pot_lines_x]]
                   for seq in seqs_x]
    seqs_y_vals = [pot_lines_y_vals[[v in seq for v in pot_lines_y]]
                   for seq in seqs_y]

    # shorten sequences to up to 9 values based on score
    # X sequences
    for i in range(len(seqs_x)):
        seq = seqs_x[i]
        seq_val = seqs_x_vals[i]

        # if the length of sequence is more than 7 + edges = 9
        # strip weakest edges
        if len(seq) > 9:
            # while not inner 7 chess lines, strip weakest edges
            while len(seq) > 7:
                if seq_val[0] > seq_val[-1]:
                    seq = seq[:-1]
                    seq_val = seq_val[:-1]
                else:
                    seq = seq[1:]
                    seq_val = seq_val[1:]

        seqs_x[i] = seq
        seqs_x_vals[i] = seq_val

    # Y sequences
    for i in range(len(seqs_y)):
        seq = seqs_y[i]
        seq_val = seqs_y_vals[i]
        while len(seq) > 9:
            if seq_val[0] > seq_val[-1]:
                seq = seq[:-1]
                seq_val = seq_val[:-1]
            else:
                seq = seq[1:]
                seq_val = seq_val[1:]

        seqs_y[i] = seq
        seqs_y_vals[i] = seq_val

    # Now that we only have length 7-9 sequences, score and choose the best one
    scores_x = np.array([np.mean(v) for v in seqs_x_vals])
    scores_y = np.array([np.mean(v) for v in seqs_y_vals])

    # Keep first sequence with the largest step size
    # scores_x = np.array([np.median(np.diff(s)) for s in seqs_x])
    # scores_y = np.array([np.median(np.diff(s)) for s in seqs_y])

    # TODO (elucidation): Choose heuristic score between step size and hough response

    best_seq_x = seqs_x[scores_x.argmax()]
    best_seq_y = seqs_y[scores_y.argmax()]
    # print(best_seq_x, best_seq_y)

    # Now if we have sequences greater than length 7, (up to 9),
    # that means we have up to 9 possible combinations of sets of 7 sequences
    # We try all of them and see which has the best checkerboard response
    sub_seqs_x = [best_seq_x[k:k+7] for k in range(len(best_seq_x) - 7 + 1)]
    sub_seqs_y = [best_seq_y[k:k+7] for k in range(len(best_seq_y) - 7 + 1)]

    dx = np.median(np.diff(best_seq_x))
    dy = np.median(np.diff(best_seq_y))
    corners = np.zeros(4, dtype=int)

    # Add 1 buffer to include the outer tiles, since sequences are only using
    # inner chessboard lines
    corners[0] = int(best_seq_y[0]-dy)
    corners[1] = int(best_seq_x[0]-dx)
    corners[2] = int(best_seq_y[-1]+dy)
    corners[3] = int(best_seq_x[-1]+dx)

    # Generate crop image with on full sequence, which may be wider than a normal
    # chessboard by an extra 2 tiles, we'll iterate over all combinations
    # (up to 9) and choose the one that correlates best with a chessboard
    gray_img_crop = PIL.Image.fromarray(img_arr_gray).crop(corners)

    # Build a kernel image of an idea chessboard to correlate against
    k = 8  # Arbitrarily chose 8x8 pixel tiles for correlation image
    quad = np.ones([k, k])
    kernel = np.vstack([np.hstack([quad, -quad]), np.hstack([-quad, quad])])
    # Becomes an 8x8 alternating grid (chessboard)
    kernel = np.tile(kernel, (4, 4))
    kernel = kernel/np.linalg.norm(kernel)  # normalize
    # 8*8 = 64x64 pixel ideal chessboard

    k = 0
    n = max(len(sub_seqs_x), len(sub_seqs_y))
    final_corners = None
    best_score = None

    # Iterate over all possible combinations of sub sequences and keep the corners
    # with the best correlation response to the ideal 64x64px chessboard
    for i in range(len(sub_seqs_x)):
        for j in range(len(sub_seqs_y)):
            k = k + 1

            # [y, x, y, x]
            sub_corners = np.array([
                sub_seqs_y[j][0]-corners[0]-dy, sub_seqs_x[i][0]-corners[1]-dx,
                sub_seqs_y[j][-1]-corners[0] +
                dy, sub_seqs_x[i][-1]-corners[1]+dx
            ], dtype=np.int)

            # Generate crop candidate, nearest pixel is fine for correlation check
            sub_img = gray_img_crop.crop(sub_corners).resize((64, 64))

            # Perform correlation score, keep running best corners as our final output
            # Use absolute since it's possible board is rotated 90 deg
            score = np.abs(np.sum(kernel * sub_img))
            if best_score is None or score > best_score:
                best_score = score
                final_corners = sub_corners + [
                    corners[0], corners[1], corners[0], corners[1]
                ]
    return final_corners


def get_chessboard_corners(img_arr, detect_corners=False):
    """ Returns a tuple of (corners, error_message)
    """
    if not detect_corners:
        # Don't try to detect corners. Assume the entire image is a board
        return (([0, 0, img_arr.shape[0], img_arr.shape[1]]), None)
    corners = detect_chessboard_corners(img_arr)
    if corners is None:
        return (None, "Failed to find corners in chessboard image")
    width = corners[2] - corners[0]
    height = corners[3] - corners[1]
    ratio = abs(1 - width / height)
    if ratio > 0.05:
        return (corners, "Invalid corners - chessboard size is not square")
    if corners[0] > 1 or corners[1] > 1:
        # TODO generalize this for chessboards positioned within images
        return (corners, "Invalid corners - (x,y) are too far from (0,0)")
    return (corners, None)


squares = chess.SquareSet(chess.BB_ALL)


def is_empty(img):
    return img.std() < 15


def validate_board(old_img, board, are_we_white):
    start, end = '', ''
    for s in squares:
        r = s >> 3
        c = s & 7
        piece = board.piece_at(s)
        b_empty = (piece == None)
        if are_we_white:
            r = 7 - r
        else:
            c = 7 - c
        if not b_empty and is_empty(get_square_img(r, c, old_img)):
            start = convert_row_column_to_square_name(r, c, are_we_white)
        if b_empty and not is_empty(get_square_img(r, c, old_img)):
            end = convert_row_column_to_square_name(r, c, are_we_white)
    if len(start) == len(end):
        return [start + end]
    if len(start) > len(end):
        print(
            f"found the starting {start} but not the ending\n please enter the ending...  ")
        end = input()
        return [start + end]
    # else:
    #     ends = []
    #     for move in board.legal_moves:
    #         if move[0:2] == start:
    #             ends.append(move[2:4])


def get_square_img(row, column, img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    r, c = img.shape
    minX = int(column * c/8)
    maxX = int((column+1)*c/8)
    minY = int(row * r/8)
    maxY = int((row + 1) * r/8)
    square = img[minY:maxY, minX:maxX]
    return square[9:-9, 9:-9]


def convert_row_column_to_square_name(row, col, are_we_white):
    if are_we_white:
        return str(chr(97 + col)) + str(8 - row)
    return str(chr(97 + (7 - col))) + str(row + 1)


def square_name_to_row_column(square, are_we_white):
    # print(square, '  from the function')
    a, b = list(square)
    if are_we_white:
        return 8 - int(b), ord(a) - 97
    return int(b) - 1, 7-ord(a) + 97


def find_possible_moves(old_img, new_img, are_we_white, board):
    starts, ends = [], []
    for i in range(8):
        for j in range(8):
            old_sq = get_square_img(i, j, old_img)
            new_sq = get_square_img(i, j, new_img)
            if cv2.absdiff(old_sq, new_sq).mean() > 8:
                if is_empty(new_sq):
                    if is_empty(old_sq):
                        continue
                    else:

                        starts.append(convert_row_column_to_square_name(
                            i, j, are_we_white))
                else:
                    ends.append(convert_row_column_to_square_name(
                        i, j, are_we_white))
    if 'e8' in starts and 'h8' in starts and 'f8' in ends and 'g8' in ends:
        return ['e8g8']
    if 'e8' in starts and 'a8' in starts and 'c8' in ends and 'd8' in ends:
        return ['e8c8']
    if 'e1' in starts and 'h1' in starts and 'g1' in ends and 'f1' in ends:
        print("they castelled....e1g1")
        return ['e1g1']
    if 'e1' in starts and 'a1' in starts and 'c1' in ends and 'd1' in ends:
        print("they castelled....e1c1")
        return ['e1c1']
    # print(starts, ends)
    moves = [''.join([a, b]) for a in starts for b in ends]

    # return moves
    # if len(moves) == 0:
    #     print('how the fuck its no move.........................')
    #     cv2.imwrite('old.jpg', old_img)
    #     cv2.imwrite('new.jpg', new_img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     exit()
    if len(moves) > 20:
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite('curr.jpg', new_img)
        cv2.imwrite('old.jpg', old_img)
        print('this is very weird')
        os._exit(1)
    valid_moves = []
    for move in moves:
        if chess.Move.from_uci(move + 'q') in board.legal_moves:
            return [move + 'q']
        if chess.Move.from_uci(move) in board.legal_moves:
            valid_moves.append(move)
    if len(valid_moves) > 1:
        # print('more than one move found finding the valid move.... resolving issue .....', valid_moves)
        # moves = []
        # for move in valid_moves:
        #     row_start, col_start = square_name_to_row_column(
        #         move[0:2], are_we_white)
        #     row_end, col_end = square_name_to_row_column(
        #         move[2:4], are_we_white)

        #     if not is_empty(get_square_img(row_start, col_start, old_img)):
        #         if not is_empty(get_square_img(row_end, col_end, new_img)):
        #             continue
        #     moves.append(move)
        print("found more than one valid move....", valid_moves)
        if len(valid_moves) == 2:
            print('using another technique to resolve the issue .......')
            last_move = board.pop()
            board.push_san(board.san(last_move))
            last_move = str(last_move)[2:4]
            return [valid_moves[0]] if valid_moves[0][2:4] != last_move else [valid_moves[1]]

        # return moves

        # cv2.imshow('old', old_img)
        # cv2.imshow('new', new_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # exit()
        # for move in valid_moves:
        #     row_start, col_start = square_name_to_row_column(
        #         move[:2], are_we_white)
        #     row_end, col_end = square_name_to_row_column(
        #         move[2:], are_we_white)
        #     old_sq = get_square_img(row_start, col_start, old_img)
        #     new_sq = get_square_img(row_end, col_end, new_img)
        #     diff = cv2.absdiff(old_sq[43:51, 78:80],
        #                        new_sq[43:51, 78:80]).mean()
        #     print('difference found between ', move, '  ', diff)
        #     if diff < 60:
        #         return [move]
        # print('unable to detect the valid more: ')

    else:
        if len(valid_moves) > 0:
            print('valid moves:', valid_moves)
        return valid_moves

    # for i in range(len(moves)):
    #     if chess.Move.from_uci(moves[i]) in board.legal_moves:
    #         row_start, col_start = square_name_to_row_column(
    #             moves[i][:2], are_we_white)
    #         row_end, col_end = square_name_to_row_column(
    #             moves[i][2:], are_we_white)
    #         old_sq = get_square_img(row_start, col_start, old_img)
    #         new_sq = get_square_img(row_end, col_end, new_img)
    #         if cv2.absdiff(old_sq[43:51, 78:80], new_sq[43:51, 78:80]).mean() < 25:
    #             return [moves[i]]


def find_square_center(square_name, are_we_white, minx, miny, maxx, maxy):
    row, column = square_name_to_row_column(
        square_name, are_we_white)
    X = int(minx + (column + 0.5) * (maxx-minx)/8)
    Y = int(miny + (row + 0.5) * (maxy-miny)/8)
    return X, Y


def board_changed(old, new):
    for i in range(8):
        for j in range(8):
            o = get_square_img(i, j, old)
            n = get_square_img(i, j, new)
            if cv2.absdiff(o, n).mean() > 5:
                return True
    return False


def play_move(move, are_we_white, board_cordinate, bit_board, old_img):
    promotion = False
    if move[-1] == 'q':
        print('promoting to queen ' + move)
        promotion = True
    start, end = move[:2], move[2:4]
    # r1, c1 = square_name_to_row_column(start, are_we_white)
    # r2, c2 = square_name_to_row_column(end, are_we_white)
    # bit_board[r1, c1] = 0
    # bit_board[r2, c2] = 1
    s1, s2 = find_square_center(
        start, are_we_white, board_cordinate[0], board_cordinate[1], board_cordinate[2], board_cordinate[3])
    e1, e2 = find_square_center(
        end, are_we_white, board_cordinate[0], board_cordinate[1], board_cordinate[2], board_cordinate[3])
    # pyautogui.moveTo(s1, s2, 0.1,  pyautogui.easeInQuad)
    pyautogui.moveTo(s1, s2)
    pyautogui.mouseDown()
    time.sleep(0.02)
    pyautogui.mouseUp()
    # # # pyautogui.move(e1, e2, duration=0.1)
    # time.sleep(0.05)
    pyautogui.moveTo(e1, e2)
    pyautogui.mouseDown()
    time.sleep(0.02)
    pyautogui.mouseUp()
    # pyautogui.click(s1, s2)
    # pyautogui.click(e1, e2)
    # pausing some time for animation to take place
    # dist = np.sqrt((e1-s1)**2 + (e2-s1)**2)
    # time.sleep((dist/30)*0.03)
    # pyautogui.dragTo(e1,e2, 0.2)
    if promotion:
        # pyautogui.mouseUp()
        time.sleep(0.01)
        pyautogui.click(e1, e2, clicks=3, interval=0.02)
        c, r = old_img.shape
        r = int(r/16)
        c = int(c/16)

        # change the co-ordinate of source and destination sqaure accordingly
        e1 -= board_cordinate[0]
        e2 -= board_cordinate[1]
        s1 -= board_cordinate[0]
        s2 -= board_cordinate[1]

        old_img[e2 - r:e2 + r, e1 - c: e1 +
                c] = old_img[s2 - r:s2 + r, s1 - c: s1 + c]
        if (98 - ord(start[0]) + int(start[1])) & 1:
            old_img[s2 - r:s2 + r, s1 - c: s1 + c] = 230
        else:
            old_img[s2 - r:s2 + r, s1 - c: s1 + c] = 90
        # increase this time to get more time to promote to queen
        print('Time to choose queen and press "q" after done..... waiting....... ')
        keyboard.wait('q')
        print('Game is being continued........:)')
        return old_img
    # return np.array(sct.grab(board_img))
    try:
        img = dshot.screenshot(region=board_cordinate).view()
        np.copyto(old_img, cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    except ValueError:
        img = dshot.screenshot()[board_cordinate[1]:board_cordinate[3],
                                 board_cordinate[0]:board_cordinate[2]].view()
        np.copyto(old_img, cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    # np.copyto(old_img, cv2.cvtColor(dshot.screenshot()[board_cordinate[1]:board_cordinate[3],
    #                     board_cordinate[0]:board_cordinate[2]], cv2.COLOR_RGB2GRAY))
    r, c = square_name_to_row_column(start, are_we_white)
    r1, c1 = square_name_to_row_column(end, are_we_white)
    while 1:
        if is_empty(get_square_img(r, c, old_img)):
            if not is_empty(get_square_img(r1, c1, old_img)):
                break
        try:
            img = dshot.screenshot(region=board_cordinate).view()
            np.copyto(old_img, cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        except ValueError:
            img = dshot.screenshot()[
                board_cordinate[1]:board_cordinate[3], board_cordinate[0]:board_cordinate[2]].view()
            np.copyto(old_img, cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        print('waiting for us to move or move ', move)
    # else:
    #     print('moving again')
    #     # pyautogui.click(s1, s2)
    #     # # time.sleep(0.5)
    #     # pyautogui.click(e1, e2)
    #     np.copyto(old_img, cv2.cvtColor(dshot.screenshot(
    ##         region=board_cordinate), cv2.COLOR_RGB2GRAY))
    #     return old_img
    return old_img


def print_board(board, are_we_white):
    if are_we_white:
        print(board, '\n\n')
    else:
        fen = board.fen()
        fen = fen.split(' ')
        fen[0] = fen[0][::-1]
        fen = ' '.join(fen)
        board = chess.Board(fen)
        print(board, '\n\n')


def print_bit_board(bit_board, are_we_white):
    if are_we_white:
        print(bit_board, '\n\n')
    else:
        print(np.flip(bit_board))


def book_move(book, board):
    try:
        b = next(book.find_all(board)).move
    except StopIteration:
        return False
    return str(b)


def no_pieces(board):
    fen = board.fen()
    fen = fen.split(' ')[0]
    n = 0
    for i in fen:
        if ord(i) in range(97, 115) or ord(i) in range(65, 83):
            n += 1
    return n


def play(board, engine, thread, hash, depth, time_control, play_by_depth, online_engine):
    '''licensing stuffs'''
    # id = str(subprocess.check_output('wmic csproduct get uuid')
    #          ).split('\\r\\n')[1].strip('\\r').strip()
    # if id == '7DF9B3C2-168F-E911-8102-80E82C904838':
    #     print('Device Verified')
    # else:
    #     print('Device not Verified.....')
    #     os._exit(1)
    # valid_time = datetime(2020, 8, 30, 0, 0, 0, 0)len
    # if datetime.now() < valid_time:
    #     print('Okay')
    # else:
    #     raise Exception("Division Error:")
    #     os._exit(1)
    engine.configure({"Hash": hash})  # amount of RAM in MB
    engine.configure({"Threads": thread})
    # engine.configure({"Contempt": 50})
    book = chess.polyglot.open_reader('Performance.bin')

    _, thresh = cv2.threshold(
        dshot.screenshot(), 169, 255, cv2.THRESH_BINARY)

    board_cordinate, o = get_chessboard_corners(
        cv2.cvtColor(thresh, cv2.COLOR_RGB2GRAY), True)
    if not o:
        print('Board not found')
        os._exit(1)

    board_cordinate[0] += 2
    board_cordinate[1] += 1
    board_cordinate = tuple(int(i) for i in board_cordinate)
    first_img = cv2.cvtColor(dshot.screenshot(), cv2.COLOR_RGB2GRAY)
    # old_img = dshot.screenshot(region=board_cordinate)
    old_img = first_img[board_cordinate[1]:board_cordinate[3],
                        board_cordinate[0]:board_cordinate[2]]
    print(board_cordinate)
    cv2.imwrite('first.jpg', old_img)
    bit_board = np.zeros((8, 8), np.uint8)
    bit_board[0:2] = 1
    bit_board[6:8] = 1
    are_we_white = get_square_img(
        0, 0, old_img).mean() < get_square_img(7, 7, old_img).mean()
    new_img, curr_img = np.array([]), np.array([])
    print("running...we are white=", are_we_white)
    start, end = '', ''
    while len(start + end) == 0:
        if are_we_white:
            our_turn = True
            break
        else:
            for i in range(2):
                for j in range(8):
                    if is_empty(get_square_img(i, j, old_img)):
                        start = convert_row_column_to_square_name(
                            i, j, are_we_white)
                        bit_board[i, j] = 0
            for i in range(2, 4):
                for j in range(8):
                    if not is_empty(get_square_img(i, j, old_img)):
                        end = convert_row_column_to_square_name(
                            i, j, are_we_white)
                        bit_board[i, j] = 1
        old_img = cv2.cvtColor(dshot.screenshot(
            region=board_cordinate), cv2.COLOR_RGB2GRAY)
    if len(start + end) == 4:
        board.push_san(board.san(chess.Move.from_uci(start + end)))
        print_board(board, are_we_white)
        ##print_bit_board(bit_board, are_we_white)
        our_turn = True
    if not are_we_white and len(start + end) != 4:
        print(f'invalid move start={start} end={end}')
        os._exit(1)

    total_moves = 0
    depth_control = 32
    while 1 and not board.is_game_over():
        if our_turn:
            # time.sleep(random.randint(0, 2))
            m = False
            if total_moves < 5:
                m = book_move(book, board)
            if m:
                old_img = play_move(
                    m, are_we_white, board_cordinate, bit_board, old_img)
                print('Book Move...', str(m))
                total_moves += 1
            else:
                if play_by_depth:
                    if online_engine:
                        try:
                            online_engine.sendall(
                                f"{board.fen()},depth,{depth}".encode('utf-8'))
                            m = online_engine.recv(128).decode('utf-8')
                            print(f"OMG server responded {m}")
                        except ConnectionRefusedError:
                            print("Couldn't connect to server will try next time")
                            online_engine = False
                    else:
                        m = str(engine.play(board, chess.engine.Limit(
                            depth=depth), ponder=True).move)
                    piece = no_pieces(board)
                    if piece < depth_control:
                        if not (piece & 1):
                            depth_control = piece
                            depth += 1
                            print('depth changed to after capture... ', depth)
                else:
                    if online_engine:
                        try:
                            online_engine.sendall(
                                f"{board.fen()},time,{time_control}".encode('utf-8'))
                            m = online_engine.recv(128).decode('utf-8')
                            print(f"OMG server responded {m}")
                        except ConnectionRefusedError:
                            print("Couldn't connect to server will try next time")
                            online_engine = False
                    else:
                        m = str(engine.play(board, chess.engine.Limit(
                            time=time_control), ponder=True).move)
                total_moves += 1
                print("Engine Move... ", m)
                old_img = play_move(
                    m, are_we_white, board_cordinate, bit_board, old_img)

            #old_img = np.array(old_img)
            if board.is_game_over():
                os._exit(1)
            # pyautogui.mouseUp()
            # old_img = cv2.cvtColor(old_img, cv2.COLOR_RGB2GRAY)
            # old_img = cv2.resize(old_img, (800, 800), interpolation=cv2.INTER_AREA)
            board.push_san(board.san(chess.Move.from_uci(m)))
            print_board(board, are_we_white)
            #print_bit_board(bit_board, are_we_white)
            our_turn = False
            # print("We moved:" + m)
        while 1 and not board.is_game_over():
            time.sleep(0.08)
            flag = 0
            # new_img = np.array(sct.grab(board_img))
            new_img = dshot.screenshot(region=board_cordinate)
            new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)
            # new_img = cv2.resize(new_img, (800, 800), interpolation=cv2.INTER_AREA)

            if board_changed(old_img, new_img):
                time.sleep(0.03)
                break
            else:
                moves = validate_board(old_img, board, are_we_white)
                if moves != None and len(moves[0]) > 0:
                    print('Found a elusive move ;)', moves)
                    flag = 1
                    break

        while 1 and not board.is_game_over():
            if flag:
                break
            time.sleep(0.09)
            curr_img = dshot.screenshot(region=board_cordinate)
            curr_img = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY)
            # curr_img = cv2.resize(curr_img, (800, 800),
            #                       interpolation=cv2.INTER_AREA)
            # cv2.imwrite('curr.jpg', curr_img)
            # cv2.imwrite('old.jpg', old_img)
            if not board_changed(curr_img, new_img):
                moves = find_possible_moves(
                    old_img, curr_img, are_we_white, board)

                if len(moves) > 0:
                    np.copyto(old_img, curr_img)
                    break
            np.copyto(new_img, curr_img)
        if not our_turn:
            for move in moves:
                if chess.Move.from_uci(move) in board.legal_moves:
                    print("They moved:" + move)
                    start, end = move[:2], move[2:4]
                    r1, c1 = square_name_to_row_column(start, are_we_white)
                    r2, c2 = square_name_to_row_column(end, are_we_white)
                    bit_board[r1, c1] = 0
                    bit_board[r2, c2] = 1
                    m = chess.Move.from_uci(move)
                    if m.promotion != None:
                        print('opponent promoting to queen.....')
                        board.push_san(board.san(m))
                    else:
                        board.push_san(board.san(m))
                    print_board(board, are_we_white)
                    #print_bit_board(bit_board, are_we_white)
                    our_turn = True
                if board.is_game_over():
                    os._exit(1)


if __name__ == '__main__':
    import socket
    ADDR = ('65.0.181.232', 6751)
    try:    
        online_engine = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        online_engine.connect(ADDR)
    except ConnectionRefusedError:
        print("Sorry couldn't connect to server")
        online_engine = False
    except e:
        print(e)
        os._exit(0)
    thread = 1
    ram = 32
    depth = 5
    play_by_depth_bool = False  # Change to true to play depthwise, False to play timewise
    time_control = 0.1
    import configparser
    engine = chess.engine.SimpleEngine.popen_uci('stockfish')
    parser = configparser.ConfigParser()
    parser.read('default.ini')
    try:
        thread = int(parser.get('Engine Default Settings', 'thread'))
        ram = int(parser.get('Engine Default Settings', 'hash'))
        depth = int(parser.get('Time Control', 'depth'))
        time_control = float(parser.get('Time Control', 'time'))
        print('found the default.ini settings loaded from there')
    except configparser.NoSectionError:
        print('Either default.ini not found or the file has syntax error..\n using default settings\n\n playing by default depth of 12')
    play(chess.Board(), engine, thread, ram,
         depth, time_control, play_by_depth_bool, online_engine)
