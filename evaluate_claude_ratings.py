#!/usr/bin/env python3
"""Evaluate Claude's performance using a rating system."""

import json
import re
import asyncio
import subprocess
from typing import List, Dict, Tuple


# Rating scale:
# 1 = Definitely not duplication (false positive)
# 2 = Probably not duplication
# 3 = Borderline/uncertain
# 4 = Probably duplication  
# 5 = Definitely duplication (true positive)

# Default threshold - ratings >= this are considered true positives
DEFAULT_THRESHOLD = 3.5


async def get_claude_rating(duplicate: Dict) -> Tuple[int, str, float]:
    """Get Claude's rating for a single duplicate."""
    
    prompt = f"""Rate this code duplication on a scale of 1-5:
1 = Definitely not duplication (false positive)
2 = Probably not duplication  
3 = Borderline/uncertain
4 = Probably duplication
5 = Definitely duplication (true positive)

Respond with ONLY: <rating> - <one-line reason>

Duplicate to analyze:
{duplicate['file1']} (lines {duplicate['lines1']}) <-> {duplicate['file2']} (lines {duplicate['lines2']})
Lines: {duplicate.get('lines', 'unknown')}
Fragment:
{duplicate['fragment'][:500]}{"..." if len(duplicate['fragment']) > 500 else ""}
"""
    
    cmd = ['claude', '--model', 'sonnet', '-p', prompt]
    
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            return duplicate['id'], f"Error: {stderr.decode()}", 0.0
            
        response = stdout.decode().strip()
        
        # Parse rating from response
        # Handle both formats: "5 - reason" and "<rating>5 - reason</rating>"
        response = response.strip()
        if response.startswith('<rating>') and response.endswith('</rating>'):
            response = response[8:-9].strip()
        
        match = re.match(r'^(\d+)\s*[-–]\s*(.+)$', response)
        if match:
            rating = int(match.group(1))
            reason = match.group(2).strip()
            return duplicate['id'], reason, float(rating)
        else:
            return duplicate['id'], f"Could not parse: {response[:60]}...", 0.0
            
    except Exception as e:
        return duplicate['id'], f"Error: {str(e)}", 0.0


async def evaluate_claude_batch(duplicates: List[Dict]) -> List[Tuple[int, str, float]]:
    """Evaluate a batch of duplicates in parallel."""
    tasks = [get_claude_rating(dup) for dup in duplicates]
    return await asyncio.gather(*tasks)


def calculate_metrics(answer_sheet: List[Dict], claude_ratings: Dict[int, float], threshold: float = DEFAULT_THRESHOLD):
    """Calculate performance metrics."""
    
    # Convert ratings to binary classifications based on threshold
    total = len(answer_sheet)
    correct = 0
    
    # For more detailed analysis
    rating_differences = []
    confusion_matrix = {
        'tp': 0,  # True positive (both agree it's duplication)
        'tn': 0,  # True negative (both agree it's not duplication)
        'fp': 0,  # False positive (Claude says yes, answer says no)
        'fn': 0   # False negative (Claude says no, answer says yes)
    }
    
    for answer in answer_sheet:
        answer_rating = answer['rating']
        claude_rating = claude_ratings.get(answer['id'], 0)
        
        # Binary classification based on threshold
        answer_positive = answer_rating >= threshold
        claude_positive = claude_rating >= threshold
        
        if answer_positive and claude_positive:
            confusion_matrix['tp'] += 1
            correct += 1
        elif not answer_positive and not claude_positive:
            confusion_matrix['tn'] += 1
            correct += 1
        elif not answer_positive and claude_positive:
            confusion_matrix['fp'] += 1
        else:  # answer_positive and not claude_positive
            confusion_matrix['fn'] += 1
            
        # Track rating differences
        rating_diff = abs(answer_rating - claude_rating)
        rating_differences.append({
            'id': answer['id'],
            'answer_rating': answer_rating,
            'claude_rating': claude_rating,
            'difference': rating_diff,
            'files': f"{answer['file1']}:{answer['lines1']} <-> {answer['file2']}:{answer['lines2']}"
        })
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    precision = confusion_matrix['tp'] / (confusion_matrix['tp'] + confusion_matrix['fp']) if (confusion_matrix['tp'] + confusion_matrix['fp']) > 0 else 0
    recall = confusion_matrix['tp'] / (confusion_matrix['tp'] + confusion_matrix['fn']) if (confusion_matrix['tp'] + confusion_matrix['fn']) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Sort by rating difference to find biggest disagreements
    rating_differences.sort(key=lambda x: x['difference'], reverse=True)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': confusion_matrix,
        'total': total,
        'correct': correct,
        'threshold': threshold,
        'rating_differences': rating_differences,
        'avg_rating_difference': sum(d['difference'] for d in rating_differences) / len(rating_differences) if rating_differences else 0
    }


async def main():
    # Load answer sheet
    with open('answer_sheet_rated.json', 'r') as f:
        answer_sheet = json.load(f)
    
    # Load actual duplicate data from jscpd report
    with open('jscpd-report.json', 'r') as f:
        jscpd_data = json.load(f)
    
    # Merge fragment data into answer sheet
    duplicates_by_id = {}
    for i, dup in enumerate(jscpd_data['duplicates'], 1):
        duplicates_by_id[i] = dup
    
    for answer in answer_sheet:
        if answer['id'] in duplicates_by_id:
            answer['fragment'] = duplicates_by_id[answer['id']]['fragment']
            answer['lines'] = duplicates_by_id[answer['id']]['lines']
    
    print(f"Evaluating Claude on {len(answer_sheet)} duplicates...")
    print(f"Using rating threshold: {DEFAULT_THRESHOLD}")
    
    # Get Claude's ratings
    claude_results = await evaluate_claude_batch(answer_sheet)
    
    # Build ratings dict
    claude_ratings = {}
    for dup_id, reason, rating in claude_results:
        claude_ratings[dup_id] = rating
        if rating == 0:
            print(f"Warning: Failed to get rating for duplicate {dup_id}: {reason}")
    
    # Calculate metrics
    metrics = calculate_metrics(answer_sheet, claude_ratings)
    
    # Display results
    print("\n" + "="*80)
    print("PERFORMANCE METRICS")
    print("="*80)
    print(f"Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall: {metrics['recall']:.2%}")
    print(f"F1 Score: {metrics['f1_score']:.2%}")
    print(f"Average rating difference: {metrics['avg_rating_difference']:.2f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives: {metrics['confusion_matrix']['tp']}")
    print(f"  True Negatives: {metrics['confusion_matrix']['tn']}")
    print(f"  False Positives: {metrics['confusion_matrix']['fp']}")
    print(f"  False Negatives: {metrics['confusion_matrix']['fn']}")
    
    print(f"\nBiggest Disagreements:")
    for i, diff in enumerate(metrics['rating_differences'][:5]):
        print(f"{i+1}. ID {diff['id']}: Answer={diff['answer_rating']}, Claude={diff['claude_rating']} (diff={diff['difference']})")
        print(f"   {diff['files']}")
    
    # Save detailed results
    with open('claude_evaluation_results.json', 'w') as f:
        json.dump({
            'metrics': metrics,
            'claude_ratings': claude_ratings,
            'threshold': DEFAULT_THRESHOLD
        }, f, indent=2)
    
    print(f"\nDetailed results saved to claude_evaluation_results.json")
    
    # Test different thresholds
    print(f"\nThreshold Analysis:")
    print("Threshold | Accuracy | Precision | Recall | F1 Score")
    print("-"*53)
    for threshold in [2.5, 3.0, 3.5, 4.0, 4.5]:
        m = calculate_metrics(answer_sheet, claude_ratings, threshold)
        print(f"  {threshold:.1f}    | {m['accuracy']:7.1%} | {m['precision']:9.1%} | {m['recall']:6.1%} | {m['f1_score']:8.1%}")


if __name__ == '__main__':
    asyncio.run(main())