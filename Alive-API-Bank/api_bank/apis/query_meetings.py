from .api import API # Assuming API is in a local .api module
# import json # Not used directly in this snippet
# import os # Not used directly in this snippet
# import datetime # Not used in this snippet, but kept from original
from typing import List, Dict, Any

class QueryMeetings(API):
    description = "This API allows users to query the meetings for any other attendee."
    input_parameters = {
        'token': {'type': 'str', 'description': "User's token."},
        'attendee': {'type': 'str', 'description': 'The attendee for which meetings are being queried for.'}
    }
    output_parameters = {
        # Updated description to reflect inclusion of meeting_id
        'meetings_info': {'type': 'list', 'description': 'A list of meetings for the requested attendee, each including its meeting_id.'},
    }
    database_name = 'Meeting' # Assuming this is used elsewhere

    def __init__(self, init_database: Dict[str, Dict[str, Any]] = None, token_checker: Any = None) -> None:
        if init_database is not None:
            self.database = init_database
        else:
            self.database = {} # Database is Dict[str, Dict[str, Any]]
        self.token_checker = token_checker # Make sure token_checker is properly initialized/passed

    # check_api_call_correctness method (as previously corrected) would go here
    def check_api_call_correctness(self, response: Dict[str, Any], groundtruth: Dict[str, Any]) -> bool:
        """
        Checks the correctness of the QueryMeetings API call by comparing the response against groundtruth.
        This version is relaxed to allow different ordering of attendees within each meeting.
        """
        # 1. Compare api_name
        # Allows response api_name to be None only if groundtruth api_name is also None.
        if response.get('api_name') != groundtruth.get('api_name'):
            if groundtruth.get('api_name') is not None: 
                return False

        # 2. Compare input fields (token and attendee)
        response_input = response.get('input', {})
        groundtruth_input = groundtruth.get('input', {})

        if not isinstance(response_input, dict) or not isinstance(groundtruth_input, dict):
            # If inputs are not dicts (unexpected), fall back to direct comparison
            if response_input != groundtruth_input:
                return False
        else:
            # Compare specific known input fields
            if response_input.get('token') != groundtruth_input.get('token'):
                return False
            if response_input.get('attendee') != groundtruth_input.get('attendee'):
                return False

        # 3. Compare exception
        if response.get('exception') != groundtruth.get('exception'):
            return False

        # 4. Compare output (list of meetings) with relaxed attendee order
        response_meetings_output = response.get('output')
        groundtruth_meetings_output = groundtruth.get('output')

        # Handle cases where outputs might not be lists (e.g., None, error object)
        if type(response_meetings_output) != type(groundtruth_meetings_output):
            return False  # Different types, e.g., one is list, other is None or dict
        
        if response_meetings_output is None and groundtruth_meetings_output is None:
            return True # Both are None, outputs match
            
        if not isinstance(response_meetings_output, list) or not isinstance(groundtruth_meetings_output, list):
            # If neither is a list (and types matched, e.g. both are dicts or scalars), compare directly
            return response_meetings_output == groundtruth_meetings_output

        # At this point, both are lists. Check length.
        if len(response_meetings_output) != len(groundtruth_meetings_output):
            return False

        # Compare each meeting in the list.
        # This assumes the order of meetings in the outer list MATTERS.
        # If the order of meetings themselves doesn't matter, these lists would need to be sorted
        # by a unique meeting identifier (e.g., meeting_id) before this loop,
        # or a more complex set-based comparison of meetings would be needed.
        
        for i in range(len(response_meetings_output)):
            res_meeting_item = response_meetings_output[i]
            gt_meeting_item = groundtruth_meetings_output[i]

            # Ensure both items are dictionaries as expected for meeting objects
            if not isinstance(res_meeting_item, dict) or not isinstance(gt_meeting_item, dict):
                if res_meeting_item != gt_meeting_item: # If not dicts, direct compare
                    return False
                continue # Both are same non-dict item, or both dicts, proceed

            # Create copies to modify for comparison (specifically for attendees)
            res_meeting_copy = res_meeting_item.copy()
            gt_meeting_copy = gt_meeting_item.copy()

            # Handle 'attendees' list specifically: sort them before comparison
            if 'attendees' in res_meeting_copy or 'attendees' in gt_meeting_copy:
                res_attendees = res_meeting_copy.get('attendees', [])
                gt_attendees = gt_meeting_copy.get('attendees', [])

                # Ensure attendees are lists and elements are stringified for robust sorting
                res_attendees_list = res_attendees if isinstance(res_attendees, list) else ([] if res_attendees is None else [str(res_attendees)])
                gt_attendees_list = gt_attendees if isinstance(gt_attendees, list) else ([] if gt_attendees is None else [str(gt_attendees)])
                
                # Convert all attendee elements to strings before sorting to avoid type errors during sort
                try:
                    sorted_res_attendees = sorted([str(att) for att in res_attendees_list])
                    sorted_gt_attendees = sorted([str(att) for att in gt_attendees_list])
                except TypeError:
                    # Should not happen if attendees are strings or numbers, but as a fallback:
                    return False # Indicate error in comparison due to un-sortable types

                res_meeting_copy['attendees'] = sorted_res_attendees
                gt_meeting_copy['attendees'] = sorted_gt_attendees
            
            # Now compare the modified meeting dictionaries
            # This compares all keys. If one dict has a key the other doesn't, they won't be equal.
            if res_meeting_copy != gt_meeting_copy:
                return False
        
        return True

    def call(self, token: str, attendee: str) -> dict:
        input_parameters = {
            'token': token,
            'attendee': attendee
        }
        try:
            meetings_list = self.query_meetings(token, attendee)
        except Exception as e:
            exception_message = str(e)
            return {'api_name': self.__class__.__name__, 'input': input_parameters, 'output': None,
                    'exception': exception_message}
        else:
            return {'api_name': self.__class__.__name__, 'input': input_parameters, 'output': meetings_list,
                    'exception': None}

    def query_meetings(self, token: str, attendee: str) -> List[Dict[str, Any]]:
        """
        Queries the meetings for a person. Each returned meeting dictionary will include its 'meeting_id'.

        Parameters:
        - token (str): The user's token.
        - attendee (str): The person for whom meetings are being queried.

        Returns:
        - meetings_found (List[Dict[str, Any]]): A list of meeting information objects
                                                 for the requested attendee. Each dictionary
                                                 includes a 'meeting_id' key.
        """
        if not self.token_checker.check_token(token): # Assuming token_checker exists
            raise ValueError("Invalid token")

        normalized_attendee_query = attendee.strip().lower()
        if not normalized_attendee_query:
            raise ValueError("Attendee name cannot be empty")

        meetings_found = []
        # Iterate over items (meeting_id, meeting_data_original) of the database
        for meeting_id, meeting_data_original in self.database.items():
            # Ensure meeting_data_original is a dictionary before proceeding
            if not isinstance(meeting_data_original, dict):
                continue # Skip malformed entries

            meeting_attendees_list = meeting_data_original.get('attendees', [])
            if not isinstance(meeting_attendees_list, list):
                continue # Skip if attendees field is not a list

            normalized_meeting_attendees = [
                current_att.strip().lower()
                for current_att in meeting_attendees_list
                if isinstance(current_att, str)  # Ensure attendee entries are strings
            ]

            if normalized_attendee_query in normalized_meeting_attendees:
                # Create a new dictionary that includes the meeting_id
                # and all original meeting data.
                # The 'username' field and others shown in your example output
                # will be included if they are part of meeting_data_original.
                meeting_with_id = {'meeting_id': meeting_id, **meeting_data_original}
                meetings_found.append(meeting_with_id)

        return meetings_found