import threading
import time
import requests

def make_request(user_id):
    print(f"[User {user_id}] Emulating user clicking 'Send Message'...")
    # Add the user_id to the message so they are distinct questions
    url = f"http://127.0.0.1:8000/chat?message=Summarize%20the%20role%20of%20the%20Computer%20Centre%20{user_id}"
    
    start = time.time()
    try:
        response = requests.post(url)
        end = time.time()
        print(f"✅ [User {user_id}] Received response in {end - start:.2f} seconds!")
    except Exception as e:
        print(f"❌ [User {user_id}] Failed to connect: {e}")

if __name__ == "__main__":
    print("🚦 Starting 3 simultaneous user requests...")
    overall_start = time.time()
    
    threads = []
    # Create 3 "Users" hitting the server at the exact same millisecond
    for i in range(1, 4):
        t = threading.Thread(target=make_request, args=(i,))
        threads.append(t)
        t.start()

    # Wait for all 3 users to get their answers
    for t in threads:
        t.join()

    overall_end = time.time()
    
    print("-" * 40)
    print(f"📊 Total time for ALL 3 requests: {overall_end - overall_start:.2f} seconds")
    print("-" * 40)
    print("If your code was BLOCKING, 3 users would take ~25 seconds.")
    print("If your code is NON-BLOCKING, 3 users will finish in ~8 seconds!")
