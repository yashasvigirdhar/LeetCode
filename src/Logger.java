import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;

class Logger {

  private Map<String, Boolean> m;
  private Queue<LogMsg> q;

  /**
   * Initialize your data structure here.
   */
  public Logger() {
    m = new HashMap<>();
    q = new LinkedList<>();
  }

  /**
   * Returns true if the message should be printed in the given timestamp, otherwise returns false.
   * If this method returns false, the message will not be printed.
   * The timestamp is in seconds granularity.
   */
  public boolean shouldPrintMessage(int timestamp, String message) {
    while (!q.isEmpty() && (timestamp - q.peek().timestamp) > 10) {
      LogMsg removedMsg = q.poll();
      m.remove(removedMsg.message);
    }
    q.offer(new LogMsg(message, timestamp));
    if (m.containsKey(message)) {
      return false;
    } else {
      m.put(message, true);
      return true;
    }
  }

  class LogMsg {
    String message;
    int timestamp;

    public LogMsg(String message, int timestamp) {
      this.message = message;
      this.timestamp = timestamp;
    }
  }
}

/**
 * Your Logger object will be instantiated and called as such:
 * Logger obj = new Logger();
 * boolean param_1 = obj.shouldPrintMessage(timestamp,message);
 */