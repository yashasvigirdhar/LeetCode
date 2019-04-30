import java.util.Arrays;
import java.util.Scanner;

public class InputOutput {

  public void readInput() {
    Scanner scanner = new Scanner(System.in);
//    String tcRead = scanner.next();
    int tc = scanner.nextInt();//Integer.parseInt(tcRead);
    for (int i = 0; i < tc; i++) {
      int n = scanner.nextInt();
      System.out.println("read : " + n);
    }
  }

  public void readStringLines() {
    Scanner scanner = new Scanner(System.in);
    while (scanner.hasNextLine()) {
      String str = scanner.nextLine();
      String[] numbers = str.split(" ");
      System.out.println(Arrays.toString(numbers));
    }
  }
}
