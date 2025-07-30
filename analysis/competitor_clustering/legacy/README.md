# Legacy Code Archive

This directory contains archived code that was part of early development phases but is no longer in active use.

## Archived Files

### `01_sector_survey_phases_1_2.py`
**Date Archived**: January 2025  
**Original Purpose**: Sector classification using EDSL surveys  
**Why Archived**: Poor use of LLM surveys and EDSL - inefficient approach for this type of classification task

#### What it did:
- Used EDSL to classify companies into sectors via multiple-choice questions
- Implemented conditional logic for "Other" sectors with confidence scoring
- Attempted skip logic and piping between questions

#### Issues with this approach:
1. **Inefficient LLM usage**: Using surveys for simple classification tasks
2. **Complex skip logic**: Over-engineered conditional flow
3. **Poor cost-effectiveness**: Multiple API calls for straightforward classification
4. **Maintenance overhead**: Complex survey logic for simple tasks

#### Lessons learned:
- EDSL surveys are better suited for complex multi-step reasoning tasks
- Simple classification should use direct model calls
- Skip logic adds unnecessary complexity for basic tasks
- Cost optimization should be considered from the start

## Moving Forward
The new approach will use direct model calls for sector classification, focusing on:
- Single API call per company
- Direct prompt engineering
- Cost-effective batch processing
- Simpler, more maintainable code 