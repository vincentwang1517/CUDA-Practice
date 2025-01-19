#include <iostream>
#include <thread>
#include <mutex>
#include <vector>

class ThreadSafeCounter
{
public:
    ThreadSafeCounter() : counter_(0) {}
    
    ~ThreadSafeCounter() = default;

    int increment()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "Thread Safe Counter " << ++counter_ << std::endl;
        return counter_;
    }

private:
    mutable std::mutex mutex_;
    int counter_;
};



int main()
{
    ThreadSafeCounter sfc;

    std::vector<std::thread> thread_pool;

    for (int i = 0; i < 5; i++) {
        thread_pool.emplace_back(std::bind(&ThreadSafeCounter::increment, &sfc));
    }

    for (auto& t : thread_pool) {
        if (t.joinable()) {
            t.join();
        }
    }

    return 0;
}