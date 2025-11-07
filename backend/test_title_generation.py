"""
Quick test script to verify title generation works correctly.
"""

import asyncio
from app.services.title_service import TitleGenerationService


async def test_title_generation():
    """Test the title generation service with sample data."""

    print("Testing Title Generation Service\n" + "="*50)

    # Initialize service
    service = TitleGenerationService()

    # Test cases
    test_cases = [
        {
            "name": "Python coding question",
            "user_message": "How do I read a CSV file in Python and convert it to a dictionary?",
            "assistant_response": "You can read a CSV file in Python using the csv module. Here's how you can convert it to a dictionary...",
        },
        {
            "name": "React question",
            "user_message": "What's the difference between useState and useReducer in React?",
            "assistant_response": "useState and useReducer are both React hooks for managing state, but they serve different purposes...",
        },
        {
            "name": "General question",
            "user_message": "Can you explain how photosynthesis works?",
            "assistant_response": "Photosynthesis is the process by which plants convert light energy into chemical energy...",
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print("-" * 50)
        print(f"User: {test_case['user_message'][:60]}...")
        print(f"Assistant: {test_case['assistant_response'][:60]}...")

        try:
            # Test concise style
            concise_title = await service.generate_title(
                user_message=test_case['user_message'],
                assistant_response=test_case['assistant_response'],
                max_length=60,
                style="concise"
            )
            print(f"✓ Concise title: {concise_title}")

            # Test descriptive style
            descriptive_title = await service.generate_title(
                user_message=test_case['user_message'],
                assistant_response=test_case['assistant_response'],
                max_length=100,
                style="descriptive"
            )
            print(f"✓ Descriptive title: {descriptive_title}")

        except Exception as e:
            print(f"✗ Error: {type(e).__name__}: {str(e)}")

    # Test fallback mechanism
    print("\n\nTest Case 4: Fallback mechanism")
    print("-" * 50)
    print("Testing fallback with empty response...")

    try:
        fallback_title = service._fallback_title(
            user_message="This is a very long user message that should be truncated properly to fit within the maximum length constraint while still being readable",
            max_length=50
        )
        print(f"✓ Fallback title: {fallback_title}")
        print(f"  Length: {len(fallback_title)} chars")

    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {str(e)}")

    print("\n" + "="*50)
    print("All tests completed!")


if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_title_generation())
