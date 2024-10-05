class Car:
    def __init__(self, registration_number, max_speed):
        self.registration_number = registration_number
        self.max_speed = max_speed
        self.current_speed = 0  # Automatically set to zero
        self.travelled_distance = 0  # Automatically set to zero

# Main program
if __name__ == "__main__":
    # Create a new car with registration number ABC-123 and max speed 142 km/h
    my_car = Car("ABC-123", 142)

    # Print out all the properties of the new car
    print(f"Registration Number: {my_car.registration_number}")
    print(f"Maximum Speed: {my_car.max_speed} km/h")
    print(f"Current Speed: {my_car.current_speed} km/h")
    print(f"Travelled Distance: {my_car.travelled_distance} km")


class Car:
    def __init__(self, registration_number, max_speed):
        self.registration_number = registration_number
        self.max_speed = max_speed
        self.current_speed = 0
        self.travelled_distance = 0

    def accelerate(self, change):

        new_speed = self.current_speed + change


        if new_speed > self.max_speed:
            self.current_speed = self.max_speed
        elif new_speed < 0:
            self.current_speed = 0
        else:
            self.current_speed = new_speed



if __name__ == "__main__":

    my_car = Car("ABC-123", 142)


    my_car.accelerate(30)  # Increase speed by 30 km/h
    my_car.accelerate(70)  # Increase speed by 70 km/h
    my_car.accelerate(50)  # Increase speed by 50 km/h


    print(f"Current Speed after acceleration: {my_car.current_speed} km/h")


    my_car.accelerate(-200)


    print(f"Final Speed after emergency brake: {my_car.current_speed} km/h")


class Car:
    def __init__(self, registration_number, max_speed):
        self.registration_number = registration_number
        self.max_speed = max_speed
        self.current_speed = 0
        self.travelled_distance = 0

    def accelerate(self, change):

        new_speed = self.current_speed + change


        if new_speed > self.max_speed:
            self.current_speed = self.max_speed
        elif new_speed < 0:
            self.current_speed = 0
        else:
            self.current_speed = new_speed

    def drive(self, hours):

        self.travelled_distance += self.current_speed * hours



if __name__ == "__main__":

    my_car = Car("ABC-123", 142)


    my_car.accelerate(30)
    my_car.accelerate(70)
    my_car.accelerate(50)


    print(f"Current Speed after acceleration: {my_car.current_speed} km/h")


    my_car.drive(1.5)


    print(f"Travelled Distance after driving for 1.5 hours: {my_car.travelled_distance} km")


    my_car.accelerate(-200)


    print(f"Final Speed after emergency brake: {my_car.current_speed} km/h")


    my_car.drive(1)


    print(f"Travelled Distance after driving for 1 hour: {my_car.travelled_distance} km")

import random


class Car:
    def __init__(self, registration_number, max_speed):
        self.registration_number = registration_number
        self.max_speed = max_speed
        self.current_speed = 0
        self.travelled_distance = 0

    def accelerate(self, change):
        new_speed = self.current_speed + change
        if new_speed > self.max_speed:
            self.current_speed = self.max_speed
        elif new_speed < 0:
            self.current_speed = 0
        else:
            self.current_speed = new_speed

    def drive(self, hours):
        self.travelled_distance += self.current_speed * hours



if __name__ == "__main__":

    cars = []
    for i in range(1, 11):
        max_speed = random.randint(100, 200)
        registration_number = f"ABC-{i}"
        cars.append(Car(registration_number, max_speed))


    race_finished = False
    hours = 0

    while not race_finished:
        hours += 1
        print(f"\n--- Hour {hours} ---")

        for car in cars:

            speed_change = random.randint(-10, 15)
            car.accelerate(speed_change)


            car.drive(1)


            if car.travelled_distance >= 10000:
                race_finished = True


    print("\n--- Race Results ---")
    print(
        f"{'Registration Number':<15} {'Max Speed (km/h)':<15} {'Current Speed (km/h)':<20} {'Travelled Distance (km)':<20}")
    print("=" * 70)

    for car in cars:
        print(f"{car.registration_number:<15} {car.max_speed:<15} {car.current_speed:<20} {car.travelled_distance:<20}")



class InsufficientFundsError(Exception):
    """Custom exception for insufficient funds."""
    pass

class NegativeValueError(Exception):
    """Custom exception for negative values."""
    pass

def main():
    try:

        balance = float(input("Enter your account balance: "))
        if balance < 0:
            raise NegativeValueError("Account balance cannot be negative.")


        withdrawal = float(input("Enter the withdrawal amount: "))
        if withdrawal < 0:
            raise NegativeValueError("Withdrawal amount cannot be negative.")

        if withdrawal > balance:
            raise InsufficientFundsError("Withdrawal amount exceeds account balance.")


        balance -= withdrawal
        print(f"Withdrawal successful! Your new balance is: {balance:.2f}")

    except ValueError:
        print("Invalid input. Please enter numeric values.")
    except InsufficientFundsError as e:
        print(e)
    except NegativeValueError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()


def write_notes():
    with open('notes.txt', 'w') as file:
        notes = input("Enter your notes (press Enter to finish):\n")
        file.write(notes + '\n')
        print("Notes written to the file.")

def read_notes():
    try:
        with open('notes.txt', 'r') as file:
            print("\nExisting Notes:")
            content = file.read()
            if content:
                print(content)
            else:
                print("No notes found.")
    except FileNotFoundError:
        print("No notes file found. Please write some notes first.")

def append_notes():
    with open('notes.txt', 'a') as file:
        notes = input("Enter additional notes (press Enter to finish):\n")
        file.write(notes + '\n')
        print("Additional notes appended to the file.")

def main():
    while True:
        print("\n--- Notes Application ---")
        print("1. Write new notes")
        print("2. Read existing notes")
        print("3. Append additional notes")
        print("4. Exit")

        choice = input("Choose an option (1-4): ")

        if choice == '1':
            write_notes()
        elif choice == '2':
            read_notes()
        elif choice == '3':
            append_notes()
        elif choice == '4':
            print("Exiting the application. ")
            break
        else:
            print("Invalid option. Please choose a valid option.")

if __name__ == "__main__":
    main()
