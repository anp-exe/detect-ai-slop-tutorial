"""
card.py — turns a Slop Score into a shareable PNG card.
(This is the "bonus" card generator — the tutorial doesn't walk through it.
 Just keep it in your project folder and slop.py will use it automatically.)
"""

import os
import platform
from PIL import Image, ImageDraw, ImageFont, ImageFilter

PINK=(255,95,162); PURPLE=(168,85,247); WHITE=(245,242,236); GREY=(168,156,182)
RED=(255,90,110); AMBER=(255,180,90); GREEN=(90,220,150); BLUE=(120,180,255)

_SYS = platform.system()
if _SYS == "Darwin":          # macOS
    SANS  = "/System/Library/Fonts/Helvetica.ttc"
    SANSR = "/System/Library/Fonts/Helvetica.ttc"
    _SYS_EMOJI = "/System/Library/Fonts/Apple Color Emoji.ttc"; _SYS_EMOJI_SIZE = 160
elif _SYS == "Windows":
    SANS  = "C:/Windows/Fonts/arialbd.ttf"
    SANSR = "C:/Windows/Fonts/arial.ttf"
    _SYS_EMOJI = "C:/Windows/Fonts/seguiemj.ttf"; _SYS_EMOJI_SIZE = 109
else:                          # Linux
    SANS  = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    SANSR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    _SYS_EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"; _SYS_EMOJI_SIZE = 109

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUNDLED_EMOJI = os.path.join(_HERE, "NotoColorEmoji.ttf")
if os.path.exists(_BUNDLED_EMOJI):
    EMOJI = _BUNDLED_EMOJI; EMOJI_SIZE = 109     # Noto's bitmap strike
else:
    EMOJI = _SYS_EMOJI; EMOJI_SIZE = _SYS_EMOJI_SIZE

def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()

def paste_emoji(img, e, x, y, size=46):
    """Render a colour emoji at the font's native strike size, then resize."""
    try:
        big = ImageFont.truetype(EMOJI, EMOJI_SIZE)
        tmp = Image.new("RGBA", (EMOJI_SIZE+20, EMOJI_SIZE+20), (0,0,0,0))
        ImageDraw.Draw(tmp).text((4,4), e, font=big, embedded_color=True)
        tmp = tmp.resize((size,size), Image.LANCZOS)
        img.paste(tmp, (x,y), tmp)
    except Exception:
        pass

def _verdict(score):
    if score >= 70: return "Certified Artisanal Slop", "🥫", RED
    if score >= 50: return "Peak LinkedIn Cringe", "💼", RED
    if score >= 30: return "Mildly Insufferable", "😬", AMBER
    if score >= 15: return "Suspiciously Normal", "🤔", BLUE
    return "An Actual Human Wrote This", "😮", GREEN

def _offense_list(sig):
    """Every real offense the signals trigger (no 'human' placeholder)."""
    out = []
    if sig.get("broetry",0) >= 0.4:
        out.append(("🍞","Broetry detected", f"{int(sig['broetry']*100)}% of lines are one-liners", PINK,(60,40,80)))
    if sig.get("buzzwords",0) >= 1:
        n=sig["buzzwords"]; out.append(("📣","Buzzword overload", f"{n} corporate buzzword{'s' if n!=1 else ''}", AMBER,(70,34,52)))
    if sig.get("closers",0) >= 1:
        out.append(("🪝","Engagement bait", "fishes for comments with a question", BLUE,(34,50,70)))
    if sig.get("antithesis",0) >= 1:
        out.append(("🔁","AI antithesis", '"it\'s not X, it\'s Y" phrasing', PURPLE,(48,38,70)))
    if sig.get("hashtags",0) >= 4:
        out.append(("#️⃣","Hashtag pileup", f"{sig['hashtags']} hashtags", PURPLE,(48,38,70)))
    if sig.get("emoji_bullets",0) >= 2:
        out.append(("✨","Emoji bullet points", f"{sig['emoji_bullets']} decorative emoji lines", GREEN,(34,54,46)))
    if sig.get("dashes",0) > 6:
        out.append(("➖","Dash connoisseur", f"{sig['dashes']} dashes, a true em-dash artisan", PURPLE,(48,38,70)))
    elif sig.get("dashes",0) > 3:
        out.append(("➖","Em-dash overload", f"{sig['dashes']} dashes, a dead AI giveaway", BLUE,(34,50,70)))
    if sig.get("anaphora",0) >= 2:
        out.append(("🔁","Anaphora spam", f"{sig['anaphora']} lines reuse the same opener", PURPLE,(48,38,70)))
    return out

def offense_count(sig):
    """How many real offense boxes the card will show."""
    return len(_offense_list(sig))

def _offenses(sig):
    out = _offense_list(sig)
    if not out:
        out.append(("🌱","Refreshingly human","no major slop signals found", GREEN,(34,54,46)))
    return out[:3]

def make_card(score, sig, out="slop_card.png"):
    W,H = 820,1020; RAD=56
    base = Image.new("RGB",(W,H),(12,10,16)); d = ImageDraw.Draw(base)
    for y in range(H):
        t=y/H
        d.line([(0,y),(W,y)],fill=(max(0,int(40-18*t+18*(1-t))),max(0,int(20-6*t)),max(0,int(48-20*t))))
    glow=Image.new("RGBA",(W,H),(0,0,0,0)); gd=ImageDraw.Draw(glow)
    gd.ellipse([-100,-120,300,280],fill=(255,95,162,40))
    gd.ellipse([W-280,H-360,W+120,H+40],fill=(168,85,247,46))
    base.paste(glow.filter(ImageFilter.GaussianBlur(90)),(0,0),glow.filter(ImageFilter.GaussianBlur(90)))
    d=ImageDraw.Draw(base)
    d.rounded_rectangle([10,10,W-11,H-11],radius=RAD,outline=PURPLE,width=5)
    d.rounded_rectangle([22,22,W-23,H-23],radius=RAD-12,outline=(70,58,92),width=2)

    d.text((54,52),"LINKEDIN SLOP DETECTOR",font=font(SANS,30),fill=WHITE)
    sub="powered by Hugging Face"; d.text((54,96),sub,font=font(SANSR,20),fill=GREY)
    paste_emoji(base,"🤗",int(54+d.textlength(sub,font=font(SANSR,20))+10),90,34)

    cx,cy,r=W//2,302,118
    v_text,v_emoji,col = _verdict(score)
    d.arc([cx-r,cy-r,cx+r,cy+r],start=130,end=410,fill=(54,44,72),width=24)
    d.arc([cx-r,cy-r,cx+r,cy+r],start=130,end=130+(410-130)*(score/100),fill=col,width=24)
    d.text((cx,cy-26),str(score),font=font(SANS,92),fill=WHITE,anchor="mm")
    d.text((cx,cy+44),"SLOP SCORE",font=font(SANS,19),fill=GREY,anchor="mm")
    vw=d.textlength(v_text,font=font(SANS,32)); px=cx-int(vw)//2-54
    d.rounded_rectangle([px,cy+92,cx+int(vw)//2+24,cy+150],radius=29,fill=(60,30,40))
    paste_emoji(base,v_emoji,int(px+14),cy+98,40)
    d.text((px+62,cy+102),v_text,font=font(SANS,32),fill=col)

    d.text((60,520),"TOP OFFENSES",font=font(SANS,20),fill=GREY)
    ty=564
    for e,name,desc,acc,tint in _offenses(sig):
        d.rounded_rectangle([60,ty,W-60,ty+98],radius=22,fill=tint,outline=acc,width=2)
        paste_emoji(base,e,86,ty+27,46)
        d.text((156,ty+22),name,font=font(SANS,26),fill=WHITE)
        d.text((156,ty+58),desc,font=font(SANSR,19),fill=(210,200,222))
        ty+=116
    d.text((60,H-58),"made with Python + Hugging Face",font=font(SANSR,19),fill=GREY)

    mask=Image.new("L",(W,H),0); ImageDraw.Draw(mask).rounded_rectangle([4,4,W-5,H-5],radius=RAD,fill=255)
    final=Image.new("RGBA",(W,H),(0,0,0,0)); final.paste(base,(0,0),mask)
    final.save(out); return out