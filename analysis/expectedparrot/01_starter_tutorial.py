"""
ExpectedParrot Starter Tutorial Test
Following the tutorial at: https://docs.expectedparrot.com/en/latest/starter_tutorial.html
"""

import edsl
from edsl import Question, Survey, Agent, Scenario
from edsl import Model

def test_basic_functionality():
    """Test basic EDSL functionality following the starter tutorial."""
    
    print("=== ExpectedParrot Starter Tutorial Test ===\n")
    
    # Step 1: Create a simple question
    print("1. Creating a simple question...")
    q = Question(
        question_name = "favorite_color",
        question_text = "What is your favorite color?",
        question_type = "free_text"
    )
    print(f"Question created: {q.question_text}\n")
    
    # Step 2: Create a survey
    print("2. Creating a survey...")
    survey = Survey(questions = [q])
    print(f"Survey created with {len(survey.questions)} question(s)\n")
    
    # Step 3: Create an agent
    print("3. Creating an agent...")
    agent = Agent(traits = {"persona": "You are a helpful assistant."})
    print(f"Agent created with persona: {agent.traits['persona']}\n")
    
    # Step 4: Create a scenario
    print("4. Creating a scenario...")
    scenario = Scenario({"context": "You are being asked about your preferences."})
    print(f"Scenario created with context: {scenario.data['context']}\n")
    
    # Step 5: Run the survey
    print("5. Running the survey...")
    try:
        # Use a free model for testing
        model = Model("gpt-4o-mini")
        results = survey.by(agent).by(scenario).by(model).run()
        
        print("Survey completed successfully!")
        print(f"Results: {results}\n")
        
        # Display the answer
        for result in results:
            print(f"Agent's favorite color: {result.answer}\n")
            
        return True
        
    except Exception as e:
        print(f"Error running survey: {e}")
        print("This might be due to API key configuration or model availability.")
        print("Let's try with a different approach...\n")
        return False

def test_without_api():
    """Test EDSL functionality without requiring API calls."""
    
    print("=== Testing EDSL Structure Without API Calls ===\n")
    
    # Create components
    q = Question(
        question_name = "company_description",
        question_text = "Describe what this company does in one sentence:",
        question_type = "free_text"
    )
    
    survey = Survey(questions = [q])
    agent = Agent(traits = {"persona": "You are a business analyst."})
    scenario = Scenario({"company_name": "Tesla", "industry": "Automotive"})
    
    print("Components created successfully:")
    print(f"- Question: {q.question_text}")
    print(f"- Survey: {len(survey.questions)} question(s)")
    print(f"- Agent: {agent.traits['persona']}")
    print(f"- Scenario: {scenario.data}\n")
    
    print("EDSL structure is working correctly!")
    return True

if __name__ == "__main__":
    print("Testing ExpectedParrot/EDSL installation...\n")
    
    # Quick test mode - set to True to skip API calls
    QUICK_TEST = False
    
    if QUICK_TEST:
        print("🔧 QUICK TEST MODE: Testing structure only (no API calls)")
        success = test_basic_functionality()
        if success:
            print("✅ EDSL structure test PASSED!")
            print("\nTo run with API calls:")
            print("1. Set QUICK_TEST = False in this script")
            print("2. Set up ExpectedParrot account or API keys")
            print("3. Run the script again")
    else:
        # Full test with API calls
        success = test_basic_functionality()
        if success:
            print("✅ EDSL test PASSED!")
        else:
            print("❌ Test failed, checking structure...")
            test_without_api()
    
    print("=== Test Complete ===") 