# --- input params ---
n = 2 # num items
k = 4 # num flavours per item
# --- input params ---

import numpy as np
box = np.ones(n) * k
hands = np.zeros(n)
last_consumed = None
score = 0
phase_2 = False

while (remaining := sum(box)):
    # Check for Phase 2 (assuming we knew $n$ and $k$ and can thus infer the current state of the box)
    if not phase_2 and not hands.any():
        # Only singles left
        if (box <= 1).all():
            phase_2 = True
        # Only one duplicate left, but it's the same as last consumed
        elif len(duplicates := np.where(box >= 2)[0]) == 1 and duplicates[0] == last_consumed:
            phase_2 = True

    item_type = np.random.choice(len(box), p=box/remaining)
    score += 1
    box[item_type] -= 1
    hands[item_type] += 1
    if (phase_2 or hands[item_type] >= 2) and item_type != last_consumed:
        hands[item_type] -= 1 # consume item
        last_consumed = item_type
        box += hands # put the rest back
        hands[:] = 0

assert not remaining and not box.any()
print("Score:", score)
