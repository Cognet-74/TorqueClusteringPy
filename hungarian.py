import numpy as np

def hungarian(Perf):
    """
    Hungarian algorithm for finding a minimum edge-weight matching.
    
    Input:
      Perf: A 2D numpy array (MxN) representing edge weights. An entry of np.inf 
            indicates no edge exists between the corresponding vertices.
    
    Outputs:
      Matching: A 2D numpy array of the same shape as Perf with ones in the 
                positions of the matching and zeros elsewhere.
      Cost: The cost (total weight) of the minimum matching.
    """
    # Initialize Matching as zeros with same shape as Perf.
    Matching = np.zeros(Perf.shape)
    
    # Condense the Performance Matrix by removing unconnected vertices.
    # num_y: number of connected entries in each column; num_x: for each row.
    num_y = np.sum(~np.isinf(Perf), axis=0)
    num_x = np.sum(~np.isinf(Perf), axis=1)
    
    # Find indices (0-indexed) of rows/columns that are not isolated.
    x_con = np.where(num_x != 0)[0]
    y_con = np.where(num_y != 0)[0]
    
    # P_size: maximum number of nonisolated rows or columns.
    P_size = max(len(x_con), len(y_con))
    # Assemble the condensed performance matrix.
    P_cond = np.zeros((P_size, P_size))
    P_cond[:len(x_con), :len(y_con)] = Perf[np.ix_(x_con, y_con)]
    if P_cond.size == 0:
        Cost = 0
        return Matching, Cost

    # Ensure that a perfect matching exists.
    # Create a copy, then set all non-inf entries to 0.
    Edge = np.copy(P_cond)
    Edge[~np.isinf(P_cond)] = 0
    cnum = min_line_cover(Edge)
    
    # Project additional vertices and edges so that a perfect matching exists.
    Pmax = np.max(P_cond[~np.isinf(P_cond)])
    P_size_new = P_size + int(cnum)
    P_cond = np.ones((P_size_new, P_size_new)) * Pmax
    P_cond[:len(x_con), :len(y_con)] = Perf[np.ix_(x_con, y_con)]
    
    # Main loop using a step number to control the procedure.
    exit_flag = True
    stepnum = 1
    while exit_flag:
        if stepnum == 1:
            P_cond, stepnum = step1(P_cond)
        elif stepnum == 2:
            r_cov, c_cov, M, stepnum = step2(P_cond)
        elif stepnum == 3:
            c_cov, stepnum = step3(M, P_size_new)
        elif stepnum == 4:
            M, r_cov, c_cov, Z_r, Z_c, stepnum = step4(P_cond, r_cov, c_cov, M)
        elif stepnum == 5:
            M, r_cov, c_cov, stepnum = step5(M, Z_r, Z_c, r_cov, c_cov)
        elif stepnum == 6:
            P_cond, stepnum = step6(P_cond, r_cov, c_cov)
        elif stepnum == 7:
            exit_flag = False
        else:
            raise ValueError("Unexpected step number: " + str(stepnum))
    
    # Remove virtual satellites/targets and uncondense the Matching.
    # MATLAB: Matching(x_con,y_con) = M(1:length(x_con),1:length(y_con))
    Matching[np.ix_(x_con, y_con)] = M[:len(x_con), :len(y_con)]
    # Cost is the sum of the weights in Perf corresponding to matching positions.
    Cost = np.sum(Perf[Matching == 1])
    
    return Matching, Cost


# ----- Internal Functions replicating the MATLAB steps -----

def step1(P_cond):
    """ STEP 1: Subtract the smallest number in each row from that row. """
    P_size = P_cond.shape[0]
    for ii in range(P_size):
        rmin = np.min(P_cond[ii, :])
        P_cond[ii, :] = P_cond[ii, :] - rmin
    stepnum = 2
    return P_cond, stepnum

def step2(P_cond):
    """
    STEP 2: For each zero in P_cond, if there is no starred zero in its row
            or column, star it and cover its row and column.
    """
    P_size = P_cond.shape[0]
    r_cov = np.zeros((P_size, 1))
    c_cov = np.zeros((P_size, 1))
    M = np.zeros((P_size, P_size))
    for ii in range(P_size):
        for jj in range(P_size):
            if (P_cond[ii, jj] == 0) and (r_cov[ii] == 0) and (c_cov[jj] == 0):
                M[ii, jj] = 1
                r_cov[ii] = 1
                c_cov[jj] = 1
    # Reinitialize cover vectors.
    r_cov = np.zeros((P_size, 1))
    c_cov = np.zeros((P_size, 1))
    stepnum = 3
    return r_cov, c_cov, M, stepnum

def step3(M, P_size):
    """
    STEP 3: Cover each column with a starred zero.
            If all columns are covered, the matching is maximum.
    """
    c_cov = np.sum(M, axis=0)  # Sum along rows gives a 1D array.
    if np.sum(c_cov) == P_size:
        stepnum = 7
    else:
        stepnum = 4
    return c_cov, stepnum

def step4(P_cond, r_cov, c_cov, M):
    """
    STEP 4: Find a noncovered zero and prime it. If there is no starred zero 
            in its row, go to step 5; otherwise, cover the row and uncover the column.
    """
    P_size = P_cond.shape[0]
    zflag = True
    Z_r = None
    Z_c = None
    while zflag:
        row = -1
        col = -1
        exit_flag = True
        ii = 0
        jj = 0
        while exit_flag:
            if (P_cond[ii, jj] == 0) and (r_cov[ii] == 0) and (c_cov[jj] == 0):
                row = ii
                col = jj
                exit_flag = False
            jj += 1
            if jj >= P_size:
                jj = 0
                ii += 1
            if ii >= P_size:
                exit_flag = False
        if row == -1:  # No uncovered zero found.
            stepnum = 6
            zflag = False
            Z_r = 0
            Z_c = 0
        else:
            M[row, col] = 2  # Prime the zero.
            # If there is a starred zero in that row:
            if np.any(M[row, :] == 1):
                r_cov[row] = 1
                # Uncover the column(s) containing the starred zero.
                zcols = np.where(M[row, :] == 1)[0]
                c_cov[zcols] = 0
            else:
                stepnum = 5
                zflag = False
                Z_r = row
                Z_c = col
    return M, r_cov, c_cov, Z_r, Z_c, stepnum

def step5(M, Z_r, Z_c, r_cov, c_cov):
    """
    STEP 5: Construct an alternating path starting from the primed zero.
            Unstar each starred zero in the path, star each primed zero,
            erase all primes, and uncover every line. Then return to Step 3.
    """
    zflag = True
    ii = 0
    # We'll use lists to store the alternating path.
    Z_r_list = [Z_r]
    Z_c_list = [Z_c]
    while zflag:
        # Find the first starred zero in the column of the current primed zero.
        col_val = Z_c_list[ii]
        rindices = np.where(M[:, col_val] == 1)[0]
        if rindices.size > 0:
            rindex = rindices[0]
            ii += 1
            Z_r_list.append(rindex)
            Z_c_list.append(col_val)  # Same column as the primed zero.
        else:
            zflag = False
            break
        # Find the primed zero in the row of the starred zero.
        row_val = Z_r_list[ii]
        cindices = np.where(M[row_val, :] == 2)[0]
        if cindices.size > 0:
            cindex = cindices[0]
        else:
            cindex = None
        ii += 1
        Z_r_list.append(row_val)
        Z_c_list.append(cindex)
    # Alternate: unstar all starred zeros and star all primed zeros in the path.
    for i in range(len(Z_r_list)):
        r_val = Z_r_list[i]
        c_val = Z_c_list[i]
        if M[r_val, c_val] == 1:
            M[r_val, c_val] = 0
        else:
            M[r_val, c_val] = 1
    # Clear covers.
    r_cov = np.zeros_like(r_cov)
    c_cov = np.zeros_like(c_cov)
    # Remove all primes.
    M[M == 2] = 0
    stepnum = 3
    return M, r_cov, c_cov, stepnum

def step6(P_cond, r_cov, c_cov):
    """
    STEP 6: Add the minimum uncovered value to every element of each covered row,
            and subtract it from every element of each uncovered column.
            Then return to Step 4.
    """
    # Find indices of uncovered rows and columns.
    a = np.where(r_cov.flatten() == 0)[0]
    b = np.where(c_cov.flatten() == 0)[0]
    if a.size == 0 or b.size == 0:
        minval = 0
    else:
        submatrix = P_cond[np.ix_(a, b)]
        minval = np.min(submatrix)
    # Add minval to every element of each covered row.
    rows_covered = np.where(r_cov.flatten() == 1)[0]
    P_cond[rows_covered, :] = P_cond[rows_covered, :] + minval
    # Subtract minval from every element of each uncovered column.
    cols_uncovered = np.where(c_cov.flatten() == 0)[0]
    P_cond[:, cols_uncovered] = P_cond[:, cols_uncovered] - minval
    stepnum = 4
    return P_cond, stepnum

def min_line_cover(Edge):
    """
    Compute the deficiency (number of additional lines needed)
    to cover all zeros in the Edge matrix.
    """
    r_cov, c_cov, M, stepnum = step2(Edge)
    c_cov, stepnum = step3(M, Edge.shape[0])
    M, r_cov, c_cov, Z_r, Z_c, stepnum = step4(Edge, r_cov, c_cov, M)
    # Deficiency = size of matrix - sum of cover vectors.
    cnum = Edge.shape[0] - np.sum(r_cov) - np.sum(c_cov)
    return cnum

# End of Hungarian algorithm code.
