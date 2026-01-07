#!/usr/bin/bash
#
# MIT License
# 
# Copyright (c) 2024 XXXXXXXXXXXX
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# provided to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

# Running the script:
# You can just call the script (bash SCRIPTNAME) and read the help instructions.
# Below are some examples on running the script as well:
#
# Example 1: bash SCPC24.sh -n 99 -k 1   -> 99
# Example 2: bash SCPC24.sh -n  1 -k 99  -> 100
# Example 3: bash SCPC24.sh --example  -> This runs example 1 from the PDF manual
# Example 4: bash SCPC24.sh -n 3 -k 2 --monte  -> Perform Monte Carlo trials using
# n and k, it will run 300 cycles then report its findings (This also will suppress 
# detailed logging).

# Rules and Examples (Taken from the PDF manual):
# Background
# In the time leading up to Christmas, you have stocked up on coffee/chocolate,
# beer or tea or Lego bricks or wrapping paper. They come in different flavours,
# sizes or colours. All are jumbled up in a box on top of your cupboard. To get
# one out, you need to reach up into the box and pull one out and you cannot
# see or feel which is which; they are all identically shaped (for beer, they’re all
# stored outdoors in the dark and you just reach out to get a random one).

# You have bought n > 1 different flavours with k > 1 of each. You
# take them one at a time as described below. You count how many
# times you take one out until the box is empty. 
# The question is, when following these rules, how many times will
# you take things out of the box to empty it?

# Phase 1
# 1. Precondition: the box is not empty.
# 2. Take things out of the box, one at a time, and hold them in your hands.
# (For the purpose of this exercise, your hands have unlimited carrying
# capacity.)
# 3. When you have two of the same flavour in your hands, consume one of
# those two and put all the others back into the box.
# 4. When the above fails, go to Phase 2.
# 5. (the no-two-same rule) if the item you’d have selected in step 3 is the
# same as you consumed at the end of the previous round, do not pick it
# but continue taking things out until 3 leads you to select a different flavour.
# Rule 5 is illustrated in Examples 2.4.1 and 2.4.2.

# Phase 2
# In phase 2, you again take things out of the box, counting the number of times
# you take something out, but you ignore rule 3.

# Example 1:

# Let n = 3, k = 2. Initially we start with the collection {A, A,B,B,C,C}.
# Starting in Phase 1 (as k > 1), we first withdraw A, then B, then C, then B (picked
# randomly but the example assumes randomness gives us this order). We consume
# B and put the rest back into the box which now contains {A, A,B,C,C}.
# The score is 4 as we’ve taken four items out, even though we’ve only consumed
# one.
# Next, we pick C, then A, then B, then C. We consume C and {A, A,B,C}
# remain. Another 4, adding to 8.
# In step 3, we take B out. We then take C, then A, then only A remains and
# we take it out and consume it. We add 4 to the score to get 12.
# Now we have {A,B,C} left in the box, and cannot pick any more things in
# Phase 1 and must go to Phase 2.
# In Phase 2, let’s say we pick A. As we had A before, we keep it in the hand
# and add 1 to the score. We now pick C and consume C and put A back into
# the box and the score is 14. At this stage, the box contains {A,B} so we can
# consume those in any order, adding two more to the score to get 16.

# Example 2
# Let n = 2, k = 3, with the set being {D,D,D,E,E,E}.
# Picking E, then E, we consume E and {D,D,D,E,E} remains, and the
# total score so far is 2.
# Next, we take E out. We then take D, then the second E. Following the
# no-two-same rule, we cannot take E and must continue. Picking D next, we
# consume D and {D,D,E,E} remains, adding 4 to the score to get 6.
# This continues until the box is empty.

# Example 3:
# 99 Bottles of Beer is an example with n = 99 and k = 1 (though k = 1 is ruled
# out for the purpose of this puzzle), and obviously without the no-two-same rule.
# The score is 99, as we take one down and pass it around 99 times.

# Initialize the box with given n (flavours) and k (copies of each flavour)
#!/bin/bash

# Initialize flags
num_trials=300
example_flag=false
monte_flag=false
n_flag=false
k_flag=false
n_value=""
k_value=""
e_value=0
m_value=0

print_usage() {
    echo ""
    echo "Usage: "
    echo "-------------------------------------------------------------"
    echo "There are 3 modes the script supports:"
    echo ""
    echo "Mode 1: Single Run with n and k"
    echo ""
    echo "bash $0 -n <digit> -k <digit>"
    echo "Where: "
    echo "  n -> Number of flavours"
    echo "  k -> Copies of each flavour"
    echo ""
    echo "Mode 2: Run Example 1 from the PDF manual:"
    echo ""
    echo "bash $0 --example"
    echo ""
    echo "Mode 3: Perform Monte Carlo trials using n and k"
    echo ""
    echo "This will run $num_trials times then calculate the average"
    echo "To Perform Monte Carlo trials:"
    echo "bash $0 --monte -n X -k Y"
    echo ""
    echo "Example run:"
    echo "bash $0 --example"
    echo "bash $0 --monte -n 3 -k 2"
    echo "bash $0 -n 3 -k 2"
    echo "-------------------------------------------------------------"
}

# Parse options
while [[ "$1" != "" ]]; do
    case $1 in
        --example)
            example_flag=true
            ;;
        --monte)
            monte_flag=true
            ;;
        -n)
            n_flag=true
            shift
            n_value=$1
            if [[ ! "$1" =~ ^[0-9]+$ ]]; then  
                echo "Error: -n must be followed by a digit"
                exit 1
            fi
            ;;
        -k)
            k_flag=true
            shift
            k_value=$1
            if [[ ! "$1" =~ ^[0-9]+$ ]]; then  
                echo "Error: -k must be followed by a digit"
                exit 1
            fi
            ;;
        *)
            echo "Error: Invalid option or arguments"
            print_usage
            exit 1
            ;;
    esac
    shift
done

# Check flags and print appropriate response
if $example_flag && ($n_flag || $k_flag); then
    echo "Error: Cannot pass --example together with -n or -k"
    print_usage
    exit 1
elif $example_flag && $monte_flag; then
    echo "Error: Cannot pass --example together with --monte"
    print_usage
    exit 1
elif [[ "$monte_flag" == "true" && (-z "$n_value" || -z "$k_value") ]]; then
    echo "Error: If --monte flag is set, both -n and -k must also be set"
    print_usage
    exit 1
elif $example_flag; then
    e_value=1
elif $monte_flag; then
    m_value=$num_trials
elif $n_flag && $k_flag; then
    echo "Checking for $n_value flavours and $k_value copies of each flavour"
else
    echo "Error: Please pass either -n X -k X, which also allows an optional --monte or"
    echo "just pass --example (without any extra options) to run Example1 from the manual"
    print_usage
    exit 1
fi

echo | awk -v n="$n_value" -v k="$k_value" -v e="$e_value" -v m="$m_value" '
# Initialize the box with given n (flavours) and k (copies of each flavour)
function initialize_box(n, k, box) {
    for (i = 1; i <= n; i++) {
        box[i] = k
    }
}

function example1_pick_random_flavour(box,n) {
    # Static variables to persist values between calls
    if (!init_done) {
        numbers = "1,2,3,2,3,1,2,3,2,3,1,1,1,3,1,2"
        split(numbers, numList, ",")  # Initialize the list
        idx = 1  # Initialize the index
        init_done = 1  # Mark initialization as done
    }

    local_val = numList[idx]  # Get the current number
    idx++  # Move to the next index
    if (idx > length(numList))  # Wrap around if we reach the end
        idx = 1
    return local_val
}

function pick_random_flavour(box, n) {
    do {
        flavour = int(rand() * n) + 1
    } while (box[flavour] <= 0)
    return flavour
}

# Visualize the current state of the box
function visualize_box(box, hand, n) {
    printf "Current box: ["
    for (i = 1; i <= n; i++) {
        if (box[i] > 0) {
            printf " [%d: %d]", i, box[i]
        }
    }
    printf " ] , hand: ["
    for (i = 1; i <= n; i++) {
        if (hand[i] > 0) {
            printf " [%d: %d]", i, hand[i]
        }
    }
    printf " ]"
    print ""
}

# Main program
{
if (m == 0) {
    rounds = 1
} else {
    rounds = m
}
for (trial = 1; trial <= rounds; trial++) {
    cmd = "date +%s%N"
    cmd | getline seed
    close(cmd)
    srand(seed)  # Set the random seed
    if (e == 1) {
        n = 3  # Number of flavours
        k = 2  # Copies of each flavour
    }
    score = 0
    delete box
    delete hand

    # Initialize the box
    initialize_box(n, k, box)
    if (m == 0) { visualize_box(box, hand, n) }

    # Phase 1: Consume pairs
    last_flavour="Nothing"
    _counter=0
    score=0
    no_two_same = 0
    while (total_items(box, n) > 0) {
        if (e == 0) {
            flavour = pick_random_flavour(box, n)
        } else {
            flavour = example1_pick_random_flavour(box, n)
        }
        hand[flavour]++
        box[flavour]--
        _counter++
        if (m == 0) { printf "Picked flavour %d, current score: %d\n", flavour, score }
        # Check for a pair
        if ((hand[flavour] == 2) && (no_two_same != 1)) {
            if (m == 0) { printf "Picked 2 of flavour %d\n", flavour }
            # Return everything in hand to the box, except discard one of the matching consumed flavour
            for (f in hand) {
                if (f == flavour) {
                    # Discard one of the consumed flavour (so only 1 remains in hand)
                    if (hand[f] > 1) {
                        hand[f] -= 1
                    } else {
                        delete hand[f]
                    }
                    if (m == 0) { printf "Consumed 1 of flavour %d\n", flavour }
                    last_flavour = flavour
                }
                box[f] += hand[f]
                if (m == 0) { printf "Returned %d units of flavour %d to the box\n", hand[f], f }
                hand[f] = 0
                } 
            score += _counter
            _counter = 0
            }
        else if ((flavour == last_flavour) && (total_items(hand, n) == 1)) {
            if (m == 0) { printf "no-two-same rule!\n" }
            no_two_same = 1
        }
        else if ((no_two_same == 1) && (flavour != last_flavour)) {
            for (f in hand) {
                if (f == last_flavour) {
                    box[f] += hand[f]
                    if (m == 0)  { printf "Returned %d units of flavour %d to the box\n", hand[f], f }
                    hand[f] = 0
                } else {
                if (m == 0) { printf "Consuming flavour %s from hand\n", f }
                    delete hand[f]
                }
            }
            no_two_same = 0
            last_flavour="Nothing"
            score += _counter
            _counter = 0
        }
        if (total_items(box, n) == 0) {
            if (m == 0) { printf "--- Phase 2: Take remaining items out of the box ---\n" }
            if (_counter > 0) {
                for (f in hand) {
                    box[f] += hand[f]
                    hand[f] = 0
                        }
                _counter = 0
            }
            if (m == 0) { visualize_box(box, hand, n) }
            break
        }
        if (m == 0) { visualize_box(box, hand, n) }
    }

    while (total_items(box, n) > 0) {
        if (e == 0) {
            flavour = pick_random_flavour(box, n)
        } else {
            flavour = example1_pick_random_flavour(box, n)
        }
        if (m == 0) { printf "Taking flavour %d\n", flavour }
        score++
        box[flavour]--
        if (m == 0) { visualize_box(box, hand, n) }
    }

    # Output the final score
    if (m == 0) {
        printf "--------------------------\n"
        printf "Final score: %d\n", score
        printf "--------------------------\n"
        printf "\n"
    }
    trial_results[trial] = score;
}
if (m > 0) {
    sum = 0
    count = 0
    for (i in trial_results) {
        sum += trial_results[i]
        trial_results_stats[trial_results[i]]++
        count++
    }
    average = sum / count
    printf "Monte Carlo Simulation Results:\n";
    printf "Number of Trials: %d\n", count;
    printf "Monte Carlo trials stats ( Result and count of result occurences): \n"
    for (result in trial_results_stats) {
        printf "- %d (found in %d full cycles)\n", result, trial_results_stats[result]
    }
    printf "Average Pulls to Achieve Target: %d\n", int(average)
}
}

# Calculate total items in the box
function total_items(box, n, total) {
    total = 0
    for (i = 1; i <= n; i++) {
        total += box[i]
    }
    return total
}
'
