class TicTacToe {

  int[] rows, cols;
  int d1, d2;
  int n;

  /**
   * Initialize your data structure here.
   */
  public TicTacToe(int n) {
    this.n = n;
    rows = new int[n];
    cols = new int[n];
    d1 = 0;
    d2 = 0;
  }

  /**
   * Player {player} makes a move at ({rows}, {cols}).
   *
   * @param row    The rows of the board.
   * @param col    The column of the board.
   * @param player The player, can be either 1 or 2.
   * @return The current winning condition, can be either:
   * 0: No one wins.
   * 1: Player 1 wins.
   * 2: Player 2 wins.
   */
  public int move(int row, int col, int player) {
    if (player == 1) {
      rows[row]++;
      cols[col]++;
      if (row == col) {
        d1++;
      }

      if (row + col == n-1) {
        d2++;
      }
      if (rows[row] == n || cols[col] == n || d1 == n || d2 == n) {
        return 1;
      }
    } else {
      rows[row]--;
      cols[col]--;
      if (row == col) {
        d1--;
      }
      if (row + col == n-1) {
        d2--;
      }
      if (rows[row] == -n || cols[col] == -n || d1 == -n || d2 == -n) {
        return 2;
      }
    }
    return 0;
  }
}


/**
 * Your TicTacToe object will be instantiated and called as such:
 * TicTacToe obj = new TicTacToe(n);
 * int param_1 = obj.move(row,col,player);
 */