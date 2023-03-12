import chess, chess.engine

board = chess.Board()
engine = chess.engine.SimpleEngine.popen_uci('stockfish')

print(engine.options)