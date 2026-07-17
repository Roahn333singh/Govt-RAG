import time
import requests

def measure_latency(question, label):
    print(f"\n[{label}] \nSending query: '{question}'")
    url = f"http://127.0.0.1:8000/chat?message={question}"
    
    start = time.time()
    try:
        response = requests.post(url)
        end = time.time()
        latency = end - start
        
        print(f"✅ Time elapsed: {latency:.3f} seconds")
        print(f"🤖 AI Answer: {response.text[:150]}...\n")
        return latency
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return 0

if __name__ == "__main__":
    print("🚀 Starting Latency Benchmark...")
    print("-" * 50)
    
    # 1. Ask a completely random, unique question to guarantee a Cache Miss
    miss_time = measure_latency(
        "Explain the exact digitization process used by the Computer Centre", 
        "TEST 1: FORCED CACHE MISS"
    )
    
    print("-" * 50)
    time.sleep(1) # Breathe for 1 second
    
    # 2. Ask a semantically similar question (different wording) to trigger Qdrant!
    hit_time = measure_latency(
        "How does the Computer Centre digitize things?", 
        "TEST 2: EXPECTED CACHE HIT"
    )

    print("-" * 50)
    print("📊 --- FINAL LATENCY REPORT ---")
    if hit_time > 0 and miss_time > 0:
        speed_factor = miss_time / hit_time
        print(f"Original Lag (Reading PDFs + Gemini): {miss_time:.2f} seconds")
        print(f"Cached Lag   (Qdrant Semantic Math):  {hit_time:.2f} seconds")
        print(f"⚡ Your cache made the app {speed_factor:.1f}x Faster!")
