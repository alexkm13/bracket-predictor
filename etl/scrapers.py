"""
Web Scrapers for BPI, SRS, TeamRankings, Tournament Results, and Barttorvik

Usage:
    python scrapers.py --source bpi
    python scrapers.py --source srs  
    python scrapers.py --source teamrankings
    python scrapers.py --source results
    python scrapers.py --source barttorvik
    python scrapers.py --source all

Each scraper saves per-season CSV immediately (cached). Re-running skips done seasons.
"""
import pandas as pd, numpy as np, time, re, argparse
from pathlib import Path
from typing import Optional
from io import StringIO

SELECTION_SUNDAYS = {
    2002:"2002-03-10",2003:"2003-03-16",2004:"2004-03-14",2005:"2005-03-13",
    2006:"2006-03-12",2007:"2007-03-11",2008:"2008-03-16",2009:"2009-03-15",
    2010:"2010-03-14",2011:"2011-03-13",2012:"2012-03-11",2013:"2013-03-17",
    2014:"2014-03-16",2015:"2015-03-15",2016:"2016-03-13",2017:"2017-03-12",
    2018:"2018-03-11",2019:"2019-03-17",2021:"2021-03-14",2022:"2022-03-13",
    2023:"2023-03-12",2024:"2024-03-17",2025:"2025-03-16",2026:"2026-03-15",
}
HEADERS = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
OUTPUT_DIR = Path("data/raw")

def _find_col(df, patterns, exact=False):
    for c in df.columns:
        for p in patterns:
            if exact and str(c).upper()==p.upper(): return c
            elif not exact and p.lower() in str(c).lower(): return c
    return None

def _clean_espn_name(n):
    if pd.isna(n): return n
    n=re.sub(r"\(\d+-\d+\)","",str(n)).strip()
    n=re.sub(r"^#?\d+\s+","",n).strip()
    return n

def _clean_sr_name(n):
    if pd.isna(n): return ""
    return re.sub(r"\s+"," ",re.sub(r"[*†‡§¶]","",str(n).replace("NCAA",""))).strip()

# ==================== BPI ====================
def scrape_bpi(season):
    from io import StringIO
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.wait import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    
    url = f"https://www.espn.com/mens-college-basketball/bpi/_/season/{season}"
    
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument(f"user-agent={HEADERS['User-Agent']}")
    
    d = webdriver.Chrome(options=opts)
    try:
        d.get(url)
        WebDriverWait(d, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table"))
        )
        time.sleep(3)
        
        while True:
            try:
                show_more = d.find_element(By.CSS_SELECTOR, "a.AnchorLink.loadMore__link")
                show_more.click()
                time.sleep(1)
            except:
                break
        
        # Get full team names from the page (not the abbreviated table)
        teams = []
        team_elements = d.find_elements(By.CSS_SELECTOR, "a.AnchorLink[href*='/mens-college-basketball/team/']")
        seen = set()
        for el in team_elements:
            name = el.text.strip()
            if name and name not in seen and len(name) > 2:
                teams.append(name)
                seen.add(name)
        
        html = d.page_source
    finally:
        d.quit()
    
    dfs = pd.read_html(StringIO(html))
    
    # Table 1 has the stats with multi-level columns
    stats = dfs[1].copy()
    
    # Flatten multi-level columns
    if isinstance(stats.columns, pd.MultiIndex):
        stats.columns = [col[-1] if 'Unnamed' in str(col[0]) else col[-1] for col in stats.columns]
    
    # Extract BPI, OFF, DEF columns
    bpi_vals = pd.to_numeric(stats["BPI"], errors="coerce")
    off_vals = pd.to_numeric(stats["OFF"], errors="coerce")
    def_vals = pd.to_numeric(stats["DEF"], errors="coerce")
    
    # Match teams to stats by row position
    # Table 0 has N rows (abbreviations), Table 1 has N+1 rows (includes header row sometimes)
    n_teams = min(len(teams), len(bpi_vals))
    
    res = pd.DataFrame({
        "team": teams[:n_teams],
        "bpi": bpi_vals.iloc[:n_teams].values,
        "bpi_off": off_vals.iloc[:n_teams].values,
        "bpi_def": def_vals.iloc[:n_teams].values,
    })
    
    res = res.dropna(subset=["bpi"])
    res["team"] = res["team"].apply(_clean_espn_name)
    
    print(f"  BPI {season}: {len(res)} teams")
    return res.reset_index(drop=True)

def scrape_all_bpi(sy=2008,ey=2026):
    od=OUTPUT_DIR/"bpi"; od.mkdir(parents=True,exist_ok=True); all_d=[]
    for s in range(sy,ey+1):
        if s==2020: continue
        cp=od/f"bpi_{s}.csv"
        if cp.exists(): print(f"  BPI {s}: cached"); all_d.append(pd.read_csv(cp)); continue
        print(f"Scraping BPI {s}...")
        try:
            df=scrape_bpi(s); df["season"]=s; df.to_csv(cp,index=False); all_d.append(df)
        except Exception as e: print(f"  FAILED: {e}")
        time.sleep(2)
    if all_d:
        c=pd.concat(all_d,ignore_index=True); c.to_csv(od/"bpi_all.csv",index=False)
        print(f"BPI total: {len(c)} rows, {c['season'].nunique()} seasons"); return c
    return pd.DataFrame()

# ==================== SRS ====================
def scrape_srs(season):
    import requests
    url=f"https://www.sports-reference.com/cbb/seasons/men/{season}-ratings.html"
    r=requests.get(url,headers=HEADERS,timeout=30); r.raise_for_status()
    tables=pd.read_html(StringIO(r.text))
    if not tables: raise ValueError(f"No tables for SRS {season}")
    df=None
    for t in tables:
        if isinstance(t.columns,pd.MultiIndex):
            t.columns=["_".join(str(x) for x in col if "Unnamed" not in str(x) and "level" not in str(x)).strip("_") for col in t.columns]
        if any("srs" in str(c).lower() for c in t.columns): df=t; break
    if df is None: df=max(tables,key=len)
    sc=None; sr=None
    for c in df.columns:
        cl=str(c).lower()
        if any(x in cl for x in ["school","team"]) and sc is None: sc=c
        if cl=="srs" or cl.endswith("_srs"): sr=c
    if sc is None:
        for c in df.columns:
            if df[c].dtype==object: sc=c; break
    if sr is None: raise ValueError(f"No SRS column. Cols: {df.columns.tolist()}")
    if sc is None: raise ValueError(f"No School column. Cols: {df.columns.tolist()}")
    res=df[[sc,sr]].copy(); res.columns=["team","srs"]
    res=res[res["team"].astype(str)!=str(sc)]
    res=res[~res["team"].astype(str).str.contains("School|Rk|Rank",na=False)]
    res["team"]=res["team"].apply(_clean_sr_name)
    res["srs"]=pd.to_numeric(res["srs"],errors="coerce")
    res=res.dropna(subset=["team","srs"]); res=res[res["team"].str.len()>0]
    print(f"  SRS {season}: {len(res)} teams"); return res.reset_index(drop=True)

def scrape_all_srs(sy=2008,ey=2026):
    od=OUTPUT_DIR/"srs"; od.mkdir(parents=True,exist_ok=True); all_d=[]
    for s in range(sy,ey+1):
        if s==2020: continue
        cp=od/f"srs_{s}.csv"
        if cp.exists(): print(f"  SRS {s}: cached"); all_d.append(pd.read_csv(cp)); continue
        print(f"Scraping SRS {s}...")
        try:
            df=scrape_srs(s); df["season"]=s; df.to_csv(cp,index=False); all_d.append(df)
        except Exception as e: print(f"  FAILED: {e}")
        time.sleep(3)
    if all_d:
        c=pd.concat(all_d,ignore_index=True); c.to_csv(od/"srs_all.csv",index=False)
        print(f"SRS total: {len(c)} rows, {c['season'].nunique()} seasons"); return c
    return pd.DataFrame()

# ==================== TEAMRANKINGS ====================
def scrape_teamrankings(date):
    url=f"https://www.teamrankings.com/ncaa-basketball/ranking/predictive-by-other/?date={date}"
    html=None
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.wait import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        opts=Options()
        opts.add_argument("--headless=new"); opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage"); opts.add_argument(f"user-agent={HEADERS['User-Agent']}")
        d=webdriver.Chrome(options=opts)
        try:
            d.get(url); WebDriverWait(d,15).until(EC.presence_of_element_located((By.TAG_NAME,"table")))
            time.sleep(2); html=d.page_source
        finally: d.quit()
        if html: print(f"  TR: fetched via Selenium")
    except ImportError: print("  Selenium not installed, trying Playwright...")
    except Exception as e: print(f"  Selenium failed ({e}), trying Playwright...")
    if html is None:
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                b=p.chromium.launch(headless=True); ctx=b.new_context(user_agent=HEADERS["User-Agent"])
                pg=ctx.new_page(); pg.goto(url,wait_until="networkidle")
                pg.wait_for_selector("table",timeout=15000); time.sleep(2); html=pg.content(); b.close()
            if html: print(f"  TR: fetched via Playwright")
        except ImportError: pass
        except Exception as e: print(f"  Playwright failed: {e}")
    if html is None:
        raise RuntimeError("Install selenium or playwright to scrape TeamRankings")
    tables=pd.read_html(StringIO(html))
    if not tables: raise ValueError(f"No tables for TR date={date}")
    df=None
    for t in tables:
        if any("rating" in str(c).lower() for c in t.columns): df=t; break
    if df is None: df=max(tables,key=len)
    tc=_find_col(df,["team"]); rc=_find_col(df,["rating","predictive"])
    if tc is None and len(df.columns)>=2: tc=df.columns[1]
    if rc is None and len(df.columns)>=3: rc=df.columns[2]
    if tc is None or rc is None: raise ValueError(f"Cannot ID columns: {df.columns.tolist()}")
    res=pd.DataFrame({"team":df[tc].astype(str).str.strip(),"tr_predictive":pd.to_numeric(df[rc],errors="coerce")})
    res=res[res["team"].str.len()>1]; res=res[~res["team"].str.lower().str.match(r"^(team|rank|#)")]
    res=res.dropna(subset=["tr_predictive"])
    print(f"  TR ({date}): {len(res)} teams"); return res.reset_index(drop=True)

def scrape_all_teamrankings(sy=2008,ey=2026):
    od=OUTPUT_DIR/"teamrankings"; od.mkdir(parents=True,exist_ok=True); all_d=[]
    for s in range(sy,ey+1):
        if s==2020: continue
        date=SELECTION_SUNDAYS.get(s)
        if not date: continue
        cp=od/f"tr_{s}.csv"
        if cp.exists(): print(f"  TR {s}: cached"); all_d.append(pd.read_csv(cp)); continue
        print(f"Scraping TR {s} (date={date})...")
        try:
            df=scrape_teamrankings(date); df["season"]=s; df.to_csv(cp,index=False); all_d.append(df)
        except Exception as e: print(f"  FAILED: {e}")
        time.sleep(5)
    if all_d:
        c=pd.concat(all_d,ignore_index=True); c.to_csv(od/"tr_all.csv",index=False)
        print(f"TR total: {len(c)} rows"); return c
    return pd.DataFrame()

# ==================== TOURNAMENT RESULTS ====================
def scrape_tournament_results(season):
    import requests
    from bs4 import BeautifulSoup
    url=f"https://www.sports-reference.com/cbb/postseason/{season}-ncaa.html"
    r=requests.get(url,headers=HEADERS,timeout=30); r.raise_for_status()
    soup=BeautifulSoup(r.text,"html.parser"); games=[]
    bracket=soup.find("div",id="bracket") or soup.find("div",id="brackets")
    if bracket:
        rounds=bracket.find_all("div",class_="round")
        rnames=["R64","R32","S16","E8","F4","CG"]
        for ri,rdiv in enumerate(rounds):
            rn=rnames[ri] if ri<len(rnames) else f"R{ri}"
            entries=[]
            for link in rdiv.find_all("a",href=re.compile(r"/cbb/schools/")):
                tn=link.get_text(strip=True); parent=link.parent; score=None
                for s in parent.find_all(string=True):
                    t=s.strip()
                    if t.isdigit() and 30<int(t)<200: score=int(t); break
                if score is None:
                    for el in parent.find_next_siblings():
                        t=el.get_text(strip=True)
                        if t.isdigit() and 30<int(t)<200: score=int(t); break
                if score is not None: entries.append({"team":tn,"score":score,"round":rn})
            for i in range(0,len(entries)-1,2):
                a,b=entries[i],entries[i+1]
                games.append({"season":season,"round":rn,"team_i":a["team"],"team_j":b["team"],
                    "score_i":a["score"],"score_j":b["score"],"margin":a["score"]-b["score"]})
    if len(games)<30:
        print(f"  Tournament {season}: bracket parse got {len(games)}, trying table fallback")
        games=[]
        try:
            for t in pd.read_html(StringIO(r.text)):
                if len(t.columns)<3: continue
                for _,row in t.iterrows():
                    vs=[str(v).strip() for v in row.values if pd.notna(v)]
                    txts=[]; nums=[]
                    for v in vs:
                        if v.isdigit() and 30<int(v)<200: nums.append(int(v))
                        elif len(v)>2 and not v.replace(".","").isdigit(): txts.append(v)
                    if len(txts)>=2 and len(nums)>=2:
                        games.append({"season":season,"round":"UNK","team_i":txts[0],"team_j":txts[1],
                            "score_i":nums[0],"score_j":nums[1],"margin":nums[0]-nums[1]})
        except: pass
    if games and games[0].get("round")=="UNK":
        rsizes=[32,16,8,4,2,1]; rns=["R64","R32","S16","E8","F4","CG"]
        if len(games)>63: rsizes=[4]+rsizes; rns=["FF"]+rns
        idx=0
        for rn,rs in zip(rns,rsizes):
            for _ in range(rs):
                if idx<len(games): games[idx]["round"]=rn; idx+=1
    if not games: raise ValueError(f"No games parsed for {season}")
    res=pd.DataFrame(games)
    res["team_i"]=res["team_i"].apply(_clean_sr_name)
    res["team_j"]=res["team_j"].apply(_clean_sr_name)
    print(f"  Tournament {season}: {len(res)} games"); return res

def scrape_all_results(sy=2008,ey=2025):
    od=OUTPUT_DIR/"game_results"; od.mkdir(parents=True,exist_ok=True); all_d=[]
    for s in range(sy,ey+1):
        if s==2020: continue
        cp=od/f"tourn_{s}.csv"
        if cp.exists(): print(f"  Tournament {s}: cached"); all_d.append(pd.read_csv(cp)); continue
        print(f"Scraping tournament {s}...")
        try:
            df=scrape_tournament_results(s); df.to_csv(cp,index=False); all_d.append(df)
        except Exception as e: print(f"  FAILED: {e}")
        time.sleep(3)
    if all_d:
        c=pd.concat(all_d,ignore_index=True); c.to_csv(od/"results_all.csv",index=False)
        print(f"Results total: {len(c)} games, {c['season'].nunique()} seasons"); return c
    return pd.DataFrame()

# ==================== BARTTORVIK ====================
def download_barttorvik(season,od=None):
    import requests
    od=od or OUTPUT_DIR/"barttorvik"; od.mkdir(parents=True,exist_ok=True)
    fp=od/f"{season}_team_results.csv"
    if fp.exists(): print(f"  BT {season}: exists"); return str(fp)
    url=f"https://barttorvik.com/{season}_team_results.csv"
    r=requests.get(url,headers=HEADERS,timeout=30); r.raise_for_status()
    fp.write_text(r.text); print(f"  BT {season}: downloaded"); return str(fp)

def extract_barttorvik(fp):
    df=pd.read_csv(fp); cols={c:c.lower().strip() for c in df.columns}; df=df.rename(columns=cols)
    tc=None
    for x in ["team","teamname","school"]:
        if x in df.columns: tc=x; break
    oe=de=em=None
    for c in df.columns:
        cc=c.replace("_","").replace(" ","")
        if cc in ["adjoe","adjustedoffensiveefficiency"]: oe=c
        elif cc in ["adjde","adjusteddefensiveefficiency"]: de=c
        elif cc in ["adjem","adjustedefficiencymargin"]: em=c
    if tc is None: raise ValueError(f"No team col: {df.columns.tolist()}")
    if oe and de: df["barttorvik_adj_em"]=pd.to_numeric(df[oe],errors="coerce")-pd.to_numeric(df[de],errors="coerce")
    elif em: df["barttorvik_adj_em"]=pd.to_numeric(df[em],errors="coerce")
    else: raise ValueError(f"No efficiency cols: {df.columns.tolist()}")
    res=df[[tc,"barttorvik_adj_em"]].copy(); res.columns=["team","barttorvik_adj_em"]
    return res.dropna(subset=["barttorvik_adj_em"]).reset_index(drop=True)

def download_all_barttorvik(sy=2008,ey=2026):
    od=OUTPUT_DIR/"barttorvik"; all_d=[]
    for s in range(sy,ey+1):
        if s==2020: continue
        ep=od/f"bt_{s}_extracted.csv"
        if ep.exists(): print(f"  BT {s}: cached"); all_d.append(pd.read_csv(ep)); continue
        try:
            fp=download_barttorvik(s,od); df=extract_barttorvik(fp); df["season"]=s
            df.to_csv(ep,index=False); all_d.append(df)
        except Exception as e: print(f"  BT {s} FAILED: {e}")
        time.sleep(1)
    if all_d:
        c=pd.concat(all_d,ignore_index=True); c.to_csv(od/"bt_all.csv",index=False)
        print(f"BT total: {len(c)} rows"); return c
    return pd.DataFrame()

# ==================== CLI ====================
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",choices=["bpi","srs","teamrankings","results","barttorvik","all"],required=True)
    p.add_argument("--start-year",type=int,default=2008)
    p.add_argument("--end-year",type=int,default=2026)
    a=p.parse_args()
    if a.source in ("bpi","all"): scrape_all_bpi(a.start_year,a.end_year)
    if a.source in ("srs","all"): scrape_all_srs(a.start_year,a.end_year)
    if a.source in ("teamrankings","all"): scrape_all_teamrankings(a.start_year,a.end_year)
    if a.source in ("results","all"): scrape_all_results(a.start_year,min(a.end_year,2025))
    if a.source in ("barttorvik","all"): download_all_barttorvik(a.start_year,a.end_year)
    print("\nDone!")