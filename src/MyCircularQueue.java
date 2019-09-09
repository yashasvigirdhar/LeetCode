class MyCircularQueue {

  int[] arr;
  int front, rear, len, capacity;

  /**
   * Initialize your data structure here. Set the size of the queue to be k.
   */
  public MyCircularQueue(int k) {
    arr = new int[k];
    len = 0;
    rear = 0;
    front = 0;
    capacity = k;
  }

  /**
   * Insert an element into the circular queue. Return true if the operation is successful.
   */
  public boolean enQueue(int value) {
    if (isFull()) {
      return false;
    }
    arr[rear] = value;
    rear = (rear + 1) % capacity;
    len++;
    return true;
  }

  /**
   * Delete an element from the circular queue. Return true if the operation is successful.
   */
  public boolean deQueue() {
    if (isEmpty()) {
      return false;
    }
    front = (front + 1) % capacity;
    len--;
    return true;
  }

  /**
   * Get the front item from the queue.
   */
  public int Front() {
    if (!isEmpty()) {
      return arr[front];
    }
    return -1;
  }

  /**
   * Get the last item from the queue.
   */
  public int Rear() {
    if (!isEmpty()) {
      return arr[(rear - 1 + capacity) % capacity];
    }
    return -1;
  }

  /**
   * Checks whether the circular queue is empty or not.
   */
  public boolean isEmpty() {
    return len == 0;
  }

  /**
   * Checks whether the circular queue is full or not.
   */
  public boolean isFull() {
    return len == capacity;
  }
}

/**
 * Your MyCircularQueue object will be instantiated and called as such:
 * MyCircularQueue obj = new MyCircularQueue(k);
 * boolean param_1 = obj.enQueue(value);
 * boolean param_2 = obj.deQueue();
 * int param_3 = obj.Front();
 * int param_4 = obj.Rear();
 * boolean param_5 = obj.isEmpty();
 * boolean param_6 = obj.isFull();
 */