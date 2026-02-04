import time
import pytest

class Database:
    def query(self):
        # Simulate a slow real database call
        print("Connecting to real database...")
        time.sleep(5)
        return "Real data: confidential production info"

def test_database_query(monkeypatch):

    # Mock function to replace the original slow method
    def mock_query(self):
        return "Mock data: for testing only"

    # Replace Database.query with mock_query during this test
    monkeypatch.setattr(Database, "query", mock_query)

    # Execute the code under test
    db = Database()
    result = db.query()

    # Verify that the mock function was used
    assert result == "Mock data: for testing only"
    print("\n--> Test passed! No 5-second wait and no real DB connection.")

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
