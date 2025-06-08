from .api import API
import json
import os
import datetime


class AddMeeting(API):
    
    description = "This API allows users to make a reservation for a meeting and store the meeting information (e.g., topic, time, location, attendees) in the database. It will fail if any attendee is already booked for an overlapping time."
    input_parameters = {
        'token': {'type': 'str', 'description': "User's token."},
        'meeting_topic': {'type': 'str', 'description': 'The title of the meeting, no more than 50 characters.'},
        'start_time': {'type': 'str',
                       'description': 'The start time of the meeting, in the pattern of %Y-%m-%d %H:%M:%S'},
        'end_time': {'type': 'str',
                     'description': 'The end time of the meeting, in the pattern of %Y-%m-%d %H:%M:%S'},
        'location': {'type': 'str',
                     'description': 'The location where the meeting to be held, no more than 100 characters.'},
        'attendees': {'type': 'list(str)',
                      'description': 'The attendees of the meeting, including names, positions and other information.'}
    }
    output_parameters = {
        'status': {'type': 'str', 'description': 'success or failed (if an exception occurs)'}
    }

    database_name = 'Meeting'

    def __init__(self, init_database=None, token_checker=None) -> None:
        if init_database is not None: # Check for None explicitly
            self.database = init_database
        else:
            self.database = {}
        self.token_checker = token_checker

    def check_api_call_correctness(self, response, groundtruth) -> bool:
        """
        Checks if the response from the API call is correct.
        (This method is not directly affected by the conflict check change, 
         but its logic for comparing meeting_topic might need review if 
         topics are also normalized before storage, though currently only attendees are).
        """
        # Ensure 'input' and 'meeting_topic' exist before trying to access/pop
        response_input = response.get('input', {})
        groundtruth_input = groundtruth.get('input', {})

        response_topic_str = response_input.get('meeting_topic', "")
        groundtruth_topic_str = groundtruth_input.get('meeting_topic', "")

        response_content = response_topic_str.split(" ")
        groundtruth_content = groundtruth_topic_str.split(" ")
        
        content_satisfied = False
        # Avoid division by zero if union is empty (though unlikely for topics)
        union_len = len(set(response_content).union(set(groundtruth_content)))
        if union_len > 0:
            if len(set(response_content).intersection(set(groundtruth_content))) / union_len > 0.5:
                content_satisfied = True
        elif not response_content and not groundtruth_content: # Both empty, considered satisfied
             content_satisfied = True


        # Create copies for comparison after popping, to not alter original response/groundtruth dicts if they are reused
        response_input_copy = response_input.copy()
        groundtruth_input_copy = groundtruth_input.copy()

        response_input_copy.pop('meeting_topic', None)
        groundtruth_input_copy.pop('meeting_topic', None)

        # Compare remaining inputs, output, and exception
        if content_satisfied and \
           response_input_copy == groundtruth_input_copy and \
           response.get('output') == groundtruth.get('output') and \
           response.get('exception') == groundtruth.get('exception'):
            return True
        else:
            return False

    def call(self, token: str, meeting_topic: str, start_time: str, end_time: str, location: str,
             attendees: list) -> dict:
        input_parameters = {
            'token': token,
            'meeting_topic': meeting_topic,
            'start_time': start_time,
            'end_time': end_time,
            'location': location,
            'attendees': attendees
        }
        try:
            # Ensure attendees is a list, even if it's empty or None from input
            attendees_list = attendees if isinstance(attendees, list) else []
            status = self.add_meeting(token, meeting_topic, start_time, end_time, location, attendees_list)
            return {'api_name': self.__class__.__name__, 'input': input_parameters, 'output': {"status": status}, 'exception': None}
        except Exception as e:
            exception_message = str(e)
            return {'api_name': self.__class__.__name__, 'input': input_parameters, 'output': {"status": "failed"}, 'exception': exception_message}

    def add_meeting(self,  token: str, meeting_topic: str, start_time: str, end_time: str, location: str,
                    attendees: list) -> str:

        if self.token_checker is None:
            raise Exception("Token checker is not configured for AddMeeting API.")
        username = self.token_checker.check_token(token) # Fails fast if token is invalid or checker missing

        if not meeting_topic or meeting_topic.strip() == "":
            raise ValueError('Meeting Topic should not be null or empty.')

        try:
            new_meeting_start_dt = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
            new_meeting_end_dt = datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
        except ValueError as e:
            raise ValueError(f"Invalid time format for start_time or end_time: {e}. Expected YYYY-MM-DD HH:MM:SS")

        if new_meeting_start_dt >= new_meeting_end_dt:
            raise ValueError("Meeting start time must be before end time.")

        # Normalize new attendees: ensure strings, strip whitespace, filter empty, use a set.
        normalized_new_attendees = set(
            str(att).strip() for att in attendees if isinstance(att, (str, int, float)) and str(att).strip()
        )
        if not normalized_new_attendees: # Optional: decide if a meeting must have attendees
             pass # Allow meetings with no specified attendees for now, or raise ValueError

        for meeting_id, existing_meeting_details in self.database.items():
            if not isinstance(existing_meeting_details, dict): 
                continue 

            existing_start_str = existing_meeting_details.get('start_time')
            existing_end_str = existing_meeting_details.get('end_time')
            
            existing_attendees_list_raw = existing_meeting_details.get('attendees', [])
            if not isinstance(existing_attendees_list_raw, list):
                 existing_attendees_list_raw = []
            
            # Normalize existing attendees from DB for comparison
            normalized_existing_attendees = set(
                str(att).strip() for att in existing_attendees_list_raw if isinstance(att, (str, int, float)) and str(att).strip()
            )

            if not existing_start_str or not existing_end_str:
                continue

            try:
                existing_meeting_start_dt = datetime.datetime.strptime(existing_start_str, '%Y-%m-%d %H:%M:%S')
                existing_meeting_end_dt = datetime.datetime.strptime(existing_end_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                # Log or handle malformed existing data; for now, skip this entry for conflict check
                continue 

            # Time overlap check: (StartA < EndB) and (StartB < EndA)
            time_overlap = (new_meeting_start_dt < existing_meeting_end_dt) and \
                           (existing_meeting_start_dt < new_meeting_end_dt)

            if time_overlap:
                common_attendees = normalized_new_attendees.intersection(normalized_existing_attendees)
                if common_attendees:
                    conflicting_attendee_names = ", ".join(sorted(list(common_attendees))) # Sorted for consistent error messages
                    existing_topic = existing_meeting_details.get('meeting_topic', 'N/A')
                    raise Exception(
                        f"Booking conflict: Attendee(s) '{conflicting_attendee_names}' is already booked "
                        f"for meeting ID '{meeting_id}' ('{existing_topic}') scheduled between "
                        f"{existing_start_str} and {existing_end_str}."
                    )
        
        valid_ids = [int(k) for k in self.database.keys() if str(k).isdigit()]
        id_now = max(valid_ids) + 1 if valid_ids else 0

        self.database[str(id_now)] = {
            'username': username,
            'meeting_topic': meeting_topic.strip(),
            'start_time': start_time,
            'end_time': end_time,
            'location': location,
            'attendees': sorted(list(normalized_new_attendees)) # Store normalized, sorted list
        }
        
        return "success"