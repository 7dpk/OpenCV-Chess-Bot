# OpenCV Python Chess Bot
![image](https://user-images.githubusercontent.com/78639550/225333635-a7de1d32-594b-428e-b7a8-884ceec23bbb.png)

>

## Introduction
This is an OpenCV-based Python chess bot that automates the playing of chess on a computer. The bot detects the chessboard on the screen using an efficient algorithm and then leverages UCI chess engines such as Stockfish or Leela to make moves. The bot uses PyAutoGUI library to interact with the chessboard and make moves. It captures screenshots of the chessboard using d3dshot library on Windows, and MSS library on Linux. Additionally, there is also a remote version of the bot that uses a cloud server to process moves and then sends them to the client.

## Dependencies
The bot requires the following libraries to be installed:

1. OpenCV
2. PyAutoGUI
3. NumPy
4. d3dshot (Windows only)
5. MSS (Linux only)
6. python-chess
7. python-socketio-client
8. You can install these libraries by running the following command:
```bash
$ pip install opencv-python pyautogui numpy d3dshot python-chess
```
## How to use the bot
1. Make sure the chess engine you want to use (e.g., Stockfish or Leela). It resides in the same folder as `bot-offline/online.py` does
2. Make sure the chess board is visible on the screen.
3. Start a cmd terminal by it's side and try:
    ```cmd
    python chess_bot.py
    ```
4. Bot will automatically detect the board, the color you are going to play and will start making the moves
5. You can edit the parameters of the bot i.e. `threads`, `ram`, `time` & `depth` from `default.ini` settings file
6. Since the promotion to the pieces differ you can tweak a bit of the code to promote to any piece and give the signal back to work so that it continues playing.

## How the bot works
The bot uses OpenCV to detect the chessboard on the screen. It first captures a screenshot of the screen using d3dshot or MSS, depending on the operating system. It then applies a series of image processing techniques to detect the corners of the chessboard.

Once the corners are detected, the bot uses PyAutoGUI to interact with the chessboard and make moves. It first captures a screenshot of the chessboard and then uses computer vision techniques to detect the pieces on the board. It then uses the python-chess library to generate a list of legal moves and selects the best move based on the evaluation of the chess engine.

In the remote version of the bot, the client sends the screenshot of the chessboard to the cloud server, which then processes the image and sends back the best move to the client.
## Video Demo
Watch the following video to see the bot in action:

![OpenCV Chess Bot Video](https://user-images.githubusercontent.com/78639550/225330750-d877a4cf-8dda-4dcf-9b6c-3c035333fe6a.mp4)

## Conclusion
This OpenCV-based Python chess bot provides a simple yet effective way to automate the playing of chess on a computer. It is easy to use and supports any UCI chess engine. Additionally, the remote version of the bot allows users to offload the computation to a cloud server, making it possible to run

## Todos
- [x] Add chessbooks to save time/resources initially.
- [X] Add Server support and processing
- [ ] Add piece recognition algorithms(AI/ML) to make playing more robust
- [ ] Improve code quality.