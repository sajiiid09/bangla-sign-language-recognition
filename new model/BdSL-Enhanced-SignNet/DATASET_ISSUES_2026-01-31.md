# Dataset Labeling & Consistency Issues Report

**Date:** January 31, 2026  
**Total Files:** 775 videos  
**Dataset:** BdSL Enhanced SignNet Raw Data

---

## Summary Statistics

- **Total Words (Classes):** 55
- **Subjects:** 2 (S01, S02)
- **Sessions:** 2 (sess01, sess02)
- **Repetitions:** 5 (rep01-rep05)
- **Expressions:** 5 (happy, negation, neutral, question, sad)
- **Expected Files per Word:** 100 (2 subjects × 2 sessions × 5 reps × 5 expressions)
- **Actual Average per Word:** 14.09 files (14% completion)

---

## 🚨 CRITICAL ISSUES

### 1. SEVERELY INCOMPLETE DATA COLLECTION (ALL WORDS)

**Every single word in the dataset is missing the majority of expected samples:**

- **Expected:** 100 samples per word (5,500 total)
- **Actual:** 775 samples total (14% completion rate)
- **Missing:** 4,725 samples (86% of expected data)

#### Words by Severity:

**CRITICAL - Less than 10 files (<10%):**
- `কোথায়`: 2/100 files (2%) ⚠️ **MOST CRITICAL**
- `বিজ্ঞান`: 8/100 files (8%)
- `বিশ্ববিদ্যালয়`: 8/100 files (8%)
- `ভূগোল`: 8/100 files (8%)

**VERY LOW - 10-12 files (10-12%):**
- `দুঃখ`: 11/100 files (11%)
- `দেখা`: 11/100 files (11%)
- `প্রশ্ন`: 11/100 files (11%)
- `খাওয়া`: 12/100 files (12%)
- `ধন্যবাদ`: 12/100 files (12%)
- `বলা`: 12/100 files (12%)

**LOW - 13-16 files (13-16%):**
- All remaining 45 words fall in this range

---

### 2. MISSING SUBJECT S01 - SESSION 02 DATA

**Subject S01 has ZERO files for Session 02:**

- **S01 - Session 01:** 231 files ✓
- **S01 - Session 02:** 0 files ❌ **COMPLETELY MISSING**
- **S02 - Session 01:** 249 files ✓
- **S02 - Session 02:** 295 files ✓

#### Impact:
- Train/test splits will be severely imbalanced
- Cannot use session-based splitting strategy
- Subject S01 provides only 29.8% of total data

---

### 3. COMPLETE S01 ABSENCE FOR 5 WORDS

**The following words have NO Subject S01 data at all:**

1. `কোথায়` (only 2 S02 files total)
2. `দেখা` (only 11 S02 files)
3. `বিজ্ঞান` (only 8 S02 files)
4. `বিশ্ববিদ্যালয়` (only 8 S02 files)
5. `ভূগোল` (only 8 S02 files)

#### Impact:
- These words lack inter-subject variability
- Single-subject data risks overfitting
- Poor generalization expected for these classes

---

### 4. EXPRESSION IMBALANCE

**Expression distribution across all videos:**

| Expression | Count | Percentage | Deviation |
|-----------|-------|-----------|-----------|
| Neutral   | 180   | 23.2%     | +14.0%    |
| Sad       | 159   | 20.5%     | +0.6%     |
| Happy     | 156   | 20.1%     | -1.3%     |
| Question  | 152   | 19.6%     | -3.8%     |
| Negation  | 128   | 16.5%     | -18.5%    |

**Issue:** Negation is significantly underrepresented (40% less than Neutral)

---

## 📊 DATA COLLECTION PATTERN ANALYSIS

### Expected vs. Actual Pattern

**Expected Collection Protocol (per word):**
```
Subject S01:
  - Session 01: rep01-rep05 × 5 expressions = 25 samples
  - Session 02: rep01-rep05 × 5 expressions = 25 samples
  Total S01: 50 samples

Subject S02:
  - Session 01: rep01-rep05 × 5 expressions = 25 samples
  - Session 02: rep01-rep05 × 5 expressions = 25 samples
  Total S02: 50 samples

Per Word Total: 100 samples
```

**Actual Pattern Observed:**
```
Subject S01:
  - Session 01: PARTIAL data (varying rep & expression coverage)
  - Session 02: NO DATA ❌
  
Subject S02:
  - Session 01: PARTIAL data (varying rep & expression coverage)
  - Session 02: PARTIAL data (varying rep & expression coverage)
```

---

## 🔍 DETAILED WORD-BY-WORD BREAKDOWN

| Word | Files | S01 | S02 | Completion % |
|------|-------|-----|-----|--------------|
| কোথায় | 2 | 0 | 2 | 2% |
| বিজ্ঞান | 8 | 0 | 8 | 8% |
| বিশ্ববিদ্যালয় | 8 | 0 | 8 | 8% |
| ভূগোল | 8 | 0 | 8 | 8% |
| দুঃখ | 11 | ✓ | ✓ | 11% |
| দেখা | 11 | 0 | 11 | 11% |
| প্রশ্ন | 11 | ✓ | ✓ | 11% |
| খাওয়া | 12 | ✓ | ✓ | 12% |
| ধন্যবাদ | 12 | ✓ | ✓ | 12% |
| বলা | 12 | ✓ | ✓ | 12% |
| অসুস্থ | 13 | ✓ | ✓ | 13% |
| আমরা | 13 | ✓ | ✓ | 13% |
| গরম | 13 | ✓ | ✓ | 13% |
| ঠান্ডা | 13 | ✓ | ✓ | 13% |
| তুমি | 13 | ✓ | ✓ | 13% |
| পড়া | 13 | ✓ | ✓ | 13% |
| লেখা | 13 | ✓ | ✓ | 13% |
| অবাক | 15 | ✓ | ✓ | 15% |
| আমি | 15 | ✓ | ✓ | 15% |
| ইতিহাস | 15 | ✓ | ✓ | 15% |
| উত্তর | 15 | ✓ | ✓ | 15% |
| কম্পিউটার | 15 | ✓ | ✓ | 15% |
| খারাপ | 15 | ✓ | ✓ | 15% |
| খুশি | 15 | ✓ | ✓ | 15% |
| গণিত | 15 | ✓ | ✓ | 15% |
| পছন্দ | 15 | ✓ | ✓ | 15% |
| পরিবেশ | 15 | ✓ | ✓ | 15% |
| পৃথিবী | 15 | ✓ | ✓ | 15% |
| বই | 15 | ✓ | ✓ | 15% |
| বন্ধু | 15 | ✓ | ✓ | 15% |
| বাংলাদেশ | 15 | ✓ | ✓ | 15% |
| বিদায় | 15 | ✓ | ✓ | 15% |
| রাগ | 15 | ✓ | ✓ | 15% |
| শরীর | 15 | ✓ | ✓ | 15% |
| শিক্ষক | 15 | ✓ | ✓ | 15% |
| সঠিক | 15 | ✓ | ✓ | 15% |
| সময় | 15 | ✓ | ✓ | 15% |
| সুন্দর | 15 | ✓ | ✓ | 15% |
| অর্থ | 16 | ✓ | ✓ | 16% |
| উদাহরণ | 16 | ✓ | ✓ | 16% |
| কবে | 16 | ✓ | ✓ | 16% |
| কাজ | 16 | ✓ | ✓ | 16% |
| কালকে | 16 | ✓ | ✓ | 16% |
| কি | 16 | ✓ | ✓ | 16% |
| কেন | 16 | ✓ | ✓ | 16% |
| কেমন | 16 | ✓ | ✓ | 16% |
| চিন্তা | 16 | ✓ | ✓ | 16% |
| থামা | 16 | ✓ | ✓ | 16% |
| ব্যাখ্যা | 16 | ✓ | ✓ | 16% |
| ভাষা | 16 | ✓ | ✓ | 16% |
| ভুল | 16 | ✓ | ✓ | 16% |
| শোনা | 16 | ✓ | ✓ | 16% |
| সকাল | 16 | ✓ | ✓ | 16% |
| সাহায্য | 16 | ✓ | ✓ | 16% |
| নাম | 18 | ✓ | ✓ | 18% |

---

## ⚠️ IMPLICATIONS FOR MODEL TRAINING

### 1. Class Imbalance
- Extreme variation in samples per class (2-18 files)
- Model will be biased toward words with more samples
- Words with <10 samples may not learn meaningful representations

### 2. Overfitting Risk
- Very low sample counts per word (average 14 samples)
- Insufficient data for deep learning models
- High risk of memorization vs. generalization

### 3. Subject Generalization
- 5 words have only 1 subject (S02)
- S01 missing Session 02 entirely
- Poor inter-subject generalization expected

### 4. Train/Val/Test Split Challenges
- Cannot use session-based splitting (S01-sess02 missing)
- Cannot use subject-based splitting (some words lack S01)
- Must use stratified random splits with extreme care

### 5. Expression-Based Features
- Negation underrepresented
- May affect non-manual marker learning
- Expression-invariant features may be compromised

---

## 📝 RECOMMENDATIONS

### IMMEDIATE ACTIONS REQUIRED:

1. **CRITICAL: Data Collection Completion**
   - Complete S01 - Session 02 for ALL 55 words
   - Complete missing samples for all words to reach at least 50/100 per word
   - Prioritize the 4 critical words with <10 samples

2. **Address Single-Subject Words**
   - Collect S01 data for: কোথায়, দেখা, বিজ্ঞান, বিশ্ববিদ্যালয়, ভূগোল

3. **Balance Expression Distribution**
   - Collect more "negation" samples to match other expressions

### FOR CURRENT TRAINING (IF PROCEEDING):

1. **Class Weighting**
   - Implement inverse frequency weighting for loss function
   - Give higher weight to underrepresented words

2. **Data Augmentation**
   - ESSENTIAL for this dataset
   - Apply aggressive augmentation (temporal, spatial, expression)
   - Consider mixup/cutmix strategies

3. **Remove/Merge Critical Classes**
   - Consider removing `কোথায়` (only 2 samples)
   - Consider merging low-sample words or creating a "rare" category

4. **Modified Splitting Strategy**
   - Use stratified random split (70/15/15)
   - Ensure each class represented in train/val/test
   - Cannot use session or subject-based splits

5. **Expectations Management**
   - Expect low accuracy (<50% overall)
   - Per-class F1 scores will vary wildly
   - Model will struggle with underrepresented classes

---

## ✅ VALIDATION CHECKS PASSED

- ✓ All filenames follow correct naming convention
- ✓ No corrupted filename formats detected
- ✓ All files properly categorized by word, subject, session, rep, expression
- ✓ No duplicate filenames found
- ✓ File extension consistency (all .mp4)

---

## CONCLUSION

The dataset has **SEVERE data collection incompleteness issues**. While the labeling convention is correct and consistent, only 14% of the expected data has been collected. This will significantly impact model performance. The primary issues are:

1. **86% of expected data missing**
2. **S01 - Session 02 completely missing**
3. **5 words have no S01 data**
4. **4 words have critically low samples (<10)**

**Recommendation:** Complete data collection before serious model training, or adjust expectations for current training to be primarily exploratory/baseline establishment.
