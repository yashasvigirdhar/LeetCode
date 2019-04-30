import java.util.Stack;

class PhoneDirectory {

  int[] map;
  private Stack<Integer> st;

  /**
   * Initialize your data structure here
   *
   * @param maxNumbers - The maximum numbers that can be stored in the phone directory.
   */
  public PhoneDirectory(int maxNumbers) {
    map = new int[maxNumbers];
    st = new Stack<>();
    for (int i = 0; i < maxNumbers; i++) {
      map[i] = 1;
      st.push(i);
    }
  }

  /**
   * Provide a number which is not assigned to anyone.
   *
   * @return - Return an available number. Return -1 if none is available.
   */
  public int get() {
    if (st.empty()) {
      return -1;
    } else {
      Integer pop = st.pop();
      map[pop] = 0;
      return pop;
    }
  }

  /**
   * Check if a number is available or not.
   */
  public boolean check(int number) {
    if (number > map.length - 1) {
      return false;
    }
    return map[number] == 1;
  }

  /**
   * Recycle or release a number.
   */
  public void release(int number) {
    if (number > map.length - 1 || map[number] == 1) {
      return;
    }
    map[number] = 1;
    st.push(number);
  }
}

/**
 * Your PhoneDirectory object will be instantiated and called as such:
 * PhoneDirectory obj = new PhoneDirectory(maxNumbers);
 * int param_1 = obj.get();
 * boolean param_2 = obj.check(number);
 * obj.release(number);
 */