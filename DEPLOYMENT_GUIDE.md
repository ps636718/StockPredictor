# StockMarket AI Terminal — Live Deployment Guide

Ye guide aapko project live karne ke 2 methods aur Render free-tier cold-start problem (sleeping) ko solve karne ke liye **100% Free Cron Setup** batata hai.

---

## Method 1: All-in-One Render Deployment (Recommended & Sabse Simple)
Aapke FastAPI backend (`backend/app.py`) mein frontend serving already configured hai. Is method se **Frontend + Backend dono ek hi URL** par live ho jayenge.

### Steps:
1. **GitHub par Repo Push karo:**
   ```bash
   git add .
   git commit -m "Add live deployment configs and IPO fixes"
   git push origin main
   ```

2. **Render.com par jao:**
   - [Render.com](https://render.com) par Sign Up / Log In karo (GitHub account se).
   - Dashboard mein **New +** button par click karo aur **Web Service** select karo.
   - Apna GitHub repository (`stock-market`) select karke connect karo.

3. **Configure Settings:**
   - **Name:** `stock-market-terminal` (ya koi bhi unique naam)
   - **Region:** Singapore ya Oregon
   - **Branch:** `main`
   - **Root Directory:** *(Isko blank chhod do)*
   - **Runtime:** `Python 3`
   - **Build Command:** 
     ```bash
     pip install --upgrade pip && pip install -r requirements.txt
     ```
   - **Start Command:** 
     ```bash
     uvicorn backend.app:app --host 0.0.0.0 --port $PORT
     ```
   - **Instance Type:** `Free`

4. **Environment Variables (Optional):**
   - `PYTHON_VERSION`: `3.11.8`
   - `ALPHATRADE_DEMO_MODE`: `0`

5. **Deploy:**
   - Click **Deploy Web Service**.
   - 2-3 minute mein build complete ho jayega aur aapko ek live URL milega:
     👉 `https://stock-market-terminal.onrender.com`
   - Is URL ko browser mein open karo — aapka frontend terminal aur backend API dono active honge!

---

## Method 2: Frontend on Vercel + Backend on Render (Separate)

Agar aapko Frontend Vercel par fast CDN ke saath host karna hai:

1. **Backend Deploy (Method 1 ke hisaab se Render par deploy karo)**. Maan lo aapka backend URL bana:
   `https://stock-backend-xyz.onrender.com`

2. **`frontend/vercel.json` update karo:**
   `frontend/vercel.json` file open karo aur `destination` mein apna Render backend URL daal do:
   ```json
   {
     "cleanUrls": true,
     "rewrites": [
       {
         "source": "/api/:path*",
         "destination": "https://stock-backend-xyz.onrender.com/api/:path*"
       }
     ]
   }
   ```

3. **Vercel.com par jao:**
   - [Vercel.com](https://vercel.com) par login karo.
   - **Add New Project** → Select `stock-market` repo.
   - **Root Directory:** Edit par click karke `frontend` select karo.
   - **Deploy** par click karo.
   - Aapko frontend URL mil jayega: `https://stock-market-yourname.vercel.app`.

---

## ⚡ Cold-Start Solution: Render ko 24/7 Awake Kaise Rakhein (100% Free)

Render free tier 15 minute inactivity ke baad container ko sleep mode mein daal deta hai. Jab recruiter ya user visit karta hai toh 30-50 second delay aata hai.

Isko solve karne ke liye **cron-job.org** (Free) use karein:

1. [cron-job.org](https://cron-job.org) par free account banao.
2. **Cronjobs** tab mein **Create Cronjob** par click karo:
   - **Title:** `Keep StockAI Alive`
   - **URL:** `https://your-app-name.onrender.com/api/health`
   - **Schedule:** `Every 10 minutes` (Execution schedule: Every 10 min)
   - **Request Method:** `GET`
3. **Save** par click karo.

✅ **Result:** cron-job.org har 10 minute mein aapke `/api/health` endpoint ko halka sa ping bhejega. Render container kabhi sleep nahi hoga aur website hamesha **instant 1-second load** degi!

---

## Resume Showcase Format

Aap apne resume mein project ko is tarah showcase kar sakte hain:

```markdown
**AlphaTrade AI — Stock Prediction & Financial Terminal**
[Live Demo: https://stock-market-terminal.onrender.com] | [GitHub: github.com/your-username/stock-market]
• Architected a multi-horizon stock forecasting terminal using FastAPI, XGBoost, and LSTM neural networks.
• Implemented 252-day rolling normalization pipeline achieving 70.67% Day-1 directional accuracy across 1,026 walk-forward validation windows.
• Built IPO intelligence engine providing automated subscription demand scoring, GMP analysis, and valuation benchmarks.
```
