import java.util.Stack;

public class MinStack {

  private Stack<Integer> s1, s2;

  /**
   * initialize your data structure here.
   */
  public MinStack() {
    s1 = new Stack<>();
    s2 = new Stack<>();
  }

  public void push(int x) {
    s1.push(x);
    if (s2.empty() || s2.peek() >= x) {
      s2.push(x);
    }
  }

  public void pop() {
    Integer pop = s1.pop();
    if (pop.equals(s2.peek())) {
      s2.pop();
    }
  }

  public int top() {
    return s1.peek();
  }

  public int getMin() {
    return s2.peek();
  }
}
