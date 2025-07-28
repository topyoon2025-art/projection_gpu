#include <iostream>
//using namespace std;

void increment_value (int *a, int N) {
    for (int i = 0; i < N; ++i){
        a[i] = a[i] + 1;
        std::cout << "Element at index a[" << i << "]: " << a[i] << std::endl;
    }

}

int main () {
    int N = 10;
    int a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    increment_value(a, N);
    return 0;
}