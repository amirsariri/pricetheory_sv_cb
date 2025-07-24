"""
LLM-based cluster validation using EDSL.
"""

import edsl
from edsl import Question, Survey, Agent, Scenario, Model
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

from .config import CompetitorSettings


def create_cluster_summary_survey():
    """Create EDSL survey for cluster summary generation."""
    
    agent = Agent(traits={
        "persona": "You are an experienced business analyst and market researcher specializing in competitive analysis and market segmentation."
    })
    
    question = Question(
        question_name="cluster_summary",
        question_text="""Based on the following cluster of companies, provide a concise summary (2-3 sentences) of the core product-market segment this cluster represents:

Cluster Companies:
{{ scenario.cluster_companies }}

Focus on:
- What type of product/service they offer
- Who their target customers are
- What market segment they serve""",
        question_type="free_text"
    )
    
    survey = Survey(questions=[question])
    return survey, agent


def create_company_fit_survey():
    """Create EDSL survey for company fit scoring."""
    
    agent = Agent(traits={
        "persona": "You are an experienced business analyst and market researcher specializing in competitive analysis and market segmentation."
    })
    
    question = Question(
        question_name="company_fit_score",
        question_text="""For the company below, rate how well it fits the core product-market segment of this cluster on a scale of 1-10:

Target Company: {{ scenario.target_company }}

Cluster Summary: {{ scenario.cluster_summary }}

Rating Scale:
1-2: Completely different market/product (not a competitor)
3-4: Somewhat related but different focus
5-6: Moderately related, some overlap
7-8: Good fit, clear competitor
9-10: Excellent fit, direct competitor

Provide your rating (1-10) and a brief explanation:""",
        question_type="free_text"
    )
    
    survey = Survey(questions=[question])
    return survey, agent


def create_cluster_quality_survey():
    """Create EDSL survey for cluster quality assessment."""
    
    agent = Agent(traits={
        "persona": "You are an experienced business analyst and market researcher specializing in competitive analysis and market segmentation."
    })
    
    question = Question(
        question_name="cluster_quality_score",
        question_text="""Rate the overall quality of this cluster on a scale of 1-10:

Cluster Companies:
{{ scenario.cluster_companies }}

Cluster Summary: {{ scenario.cluster_summary }}

Consider:
- How cohesive are the companies?
- Do they serve the same market?
- Are they direct competitors?
- Is the cluster too broad or too narrow?

Rating Scale:
1-2: Poor cluster - companies don't belong together
3-4: Weak cluster - limited competitive relationship
5-6: Fair cluster - some competitive overlap
7-8: Good cluster - clear competitive group
9-10: Excellent cluster - strong competitive relationship

Provide your rating (1-10) and brief explanation:""",
        question_type="free_text"
    )
    
    survey = Survey(questions=[question])
    return survey, agent


def extract_numeric_score(text: str) -> float:
    """Extract numeric score from LLM response."""
    import re
    
    # Look for numbers 1-10 in the text
    numbers = re.findall(r'\b([1-9]|10)\b', text)
    if numbers:
        return float(numbers[0])
    
    # If no clear number, try to infer from text
    text_lower = text.lower()
    if any(word in text_lower for word in ['excellent', 'perfect', '10', 'nine', '9']):
        return 9.0
    elif any(word in text_lower for word in ['very good', 'great', '8', 'eight']):
        return 8.0
    elif any(word in text_lower for word in ['good', '7', 'seven']):
        return 7.0
    elif any(word in text_lower for word in ['fair', 'moderate', '6', 'six', '5', 'five']):
        return 6.0
    elif any(word in text_lower for word in ['weak', 'poor', '4', 'four', '3', 'three']):
        return 4.0
    elif any(word in text_lower for word in ['bad', 'terrible', '2', 'two', '1', 'one']):
        return 2.0
    
    return 5.0  # Default neutral score


def validate_clusters_with_llm(
    df: pd.DataFrame,
    labels: np.ndarray,
    n_clusters_to_validate: int = 10,
    model_name: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Validate clusters using LLM analysis.
    
    Args:
        df: Company dataframe with cluster assignments
        labels: Cluster labels
        n_clusters_to_validate: Number of clusters to validate
        model_name: EDSL model to use
        
    Returns:
        Dictionary with validation results
    """
    logger.info(f"Starting LLM validation of {n_clusters_to_validate} clusters")
    
    # Create surveys and agent
    summary_survey, summary_agent = create_cluster_summary_survey()
    fit_survey, fit_agent = create_company_fit_survey()
    quality_survey, quality_agent = create_cluster_quality_survey()
    model = Model(model_name)
    
    # Select clusters to validate (skip single-member clusters)
    cluster_sizes = [(label, np.sum(labels == label)) for label in np.unique(labels)]
    multi_member_clusters = [(label, size) for label, size in cluster_sizes if size > 1]
    
    if not multi_member_clusters:
        logger.warning("No multi-member clusters found for validation")
        return validation_results
    
    if len(multi_member_clusters) <= n_clusters_to_validate:
        clusters_to_validate = [label for label, _ in multi_member_clusters]
    else:
        # Prioritize larger clusters and some random smaller ones
        multi_member_clusters.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 70% large clusters + 30% random smaller clusters
        n_large = int(n_clusters_to_validate * 0.7)
        n_small = n_clusters_to_validate - n_large
        
        large_clusters = [label for label, _ in multi_member_clusters[:n_large]]
        small_clusters = np.random.choice(
            [label for label, _ in multi_member_clusters[n_large:]], 
            size=min(n_small, len(multi_member_clusters[n_large:])), 
            replace=False
        )
        clusters_to_validate = list(large_clusters) + list(small_clusters)
    
    validation_results = {
        'cluster_summaries': {},
        'company_fit_scores': {},
        'cluster_quality_scores': {},
        'overall_metrics': {}
    }
    
    # Validate each cluster
    for cluster_id in clusters_to_validate:
        logger.info(f"Validating cluster {cluster_id}")
        
        # Get companies in this cluster
        cluster_mask = labels == cluster_id
        cluster_companies = df[cluster_mask]
        
        # Create cluster summary
        cluster_text = []
        for _, company in cluster_companies.iterrows():
            company_text = f"- {company['company_name']}: {company['main_product']} (Customers: {company['main_customers']})"
            cluster_text.append(company_text)
        
        cluster_companies_text = "\n".join(cluster_text)
        
        # Step 1: Get cluster summary
        scenario1 = Scenario({
            "cluster_companies": cluster_companies_text
        })
        
        try:
            # Add job description for cluster summary
            job_description = f"Cluster Validation - Summary - Cluster {cluster_id}"
            result1 = summary_survey.by(summary_agent).by(scenario1).by(model).run(
                remote_inference_description=job_description
            )
            cluster_summary = result1[0].answer['cluster_summary']
            validation_results['cluster_summaries'][cluster_id] = cluster_summary
            
            # Step 2: Get cluster quality score
            scenario2 = Scenario({
                "cluster_companies": cluster_companies_text,
                "cluster_summary": cluster_summary
            })
            
            # Add job description for cluster quality
            job_description = f"Cluster Validation - Quality - Cluster {cluster_id}"
            result2 = quality_survey.by(quality_agent).by(scenario2).by(model).run(
                remote_inference_description=job_description
            )
            quality_response = result2[0].answer['cluster_quality_score']
            quality_score = extract_numeric_score(quality_response)
            validation_results['cluster_quality_scores'][cluster_id] = {
                'score': quality_score,
                'explanation': quality_response
            }
            
            # Step 3: Validate each company in the cluster
            company_scores = {}
            for _, company in cluster_companies.iterrows():
                company_text = f"Name: {company['company_name']}\nProduct: {company['main_product']}\nCustomers: {company['main_customers']}\nDescription: {company.get('description', 'N/A')}"
                
                scenario3 = Scenario({
                    "target_company": company_text,
                    "cluster_summary": cluster_summary
                })
                
                # Add job description for company fit
                job_description = f"Cluster Validation - Company Fit - Cluster {cluster_id} - {company['company_name']}"
                result3 = fit_survey.by(fit_agent).by(scenario3).by(model).run(
                    remote_inference_description=job_description
                )
                fit_response = result3[0].answer['company_fit_score']
                fit_score = extract_numeric_score(fit_response)
                
                company_scores[company['company_name']] = {
                    'score': fit_score,
                    'explanation': fit_response
                }
            
            validation_results['company_fit_scores'][cluster_id] = company_scores
            
        except Exception as e:
            logger.error(f"Error validating cluster {cluster_id}: {e}")
            continue
    
    # Calculate overall metrics
    quality_scores = [data['score'] for data in validation_results['cluster_quality_scores'].values()]
    if quality_scores:
        validation_results['overall_metrics'] = {
            'avg_cluster_quality': np.mean(quality_scores),
            'median_cluster_quality': np.median(quality_scores),
            'clusters_validated': len(quality_scores),
            'high_quality_clusters': sum(1 for score in quality_scores if score >= 7),
            'low_quality_clusters': sum(1 for score in quality_scores if score <= 4)
        }
    
    logger.info(f"LLM validation completed. Average cluster quality: {validation_results['overall_metrics'].get('avg_cluster_quality', 0):.2f}")
    
    return validation_results


def save_llm_validation_results(
    results: Dict[str, Any],
    output_dir: Path
) -> None:
    """Save LLM validation results to files."""
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {str(k): convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        elif hasattr(obj, 'item'):  # numpy types
            return obj.item()
        else:
            return obj
    
    # Convert results
    results_serializable = convert_numpy_types(results)
    
    # Save detailed results
    import json
    with open(output_dir / "llm_validation_results.json", 'w') as f:
        json.dump(results_serializable, f, indent=2, default=str)
    
    # Create summary CSV
    summary_rows = []
    for cluster_id, quality_data in results['cluster_quality_scores'].items():
        company_scores = results['company_fit_scores'].get(cluster_id, {})
        avg_company_score = np.mean([data['score'] for data in company_scores.values()]) if company_scores else 0
        
        summary_rows.append({
            'cluster_id': cluster_id,
            'cluster_quality_score': quality_data['score'],
            'avg_company_fit_score': avg_company_score,
            'n_companies': len(company_scores),
            'cluster_summary': results['cluster_summaries'].get(cluster_id, 'N/A')
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "llm_validation_summary.csv", index=False)
    
    logger.info(f"LLM validation results saved to {output_dir}") 