def check_maximum_error(c_out_path, k_out_path):
    # Check maximum error
    c_out = open(c_out_path, 'r')
    k_out = open(k_out_path, 'r')

    c_output_line = c_out.readline()
    c = c_output_line.split()

    k_output_line = k_out.readline()
    k = k_output_line.split()


    maximum = -1.0
    c_max = 0
    k_max = 0

    print
    "len k :  ", len(k)
    print
    "len c :  ", len(c)

    for i in range(len(c)):

        k_num = float(k[i])
        c_num = float(c[i])
        if c_num > k_num:
            if k_num == 0.0:
                error = c_num - k_num
            else:
                error = (c_num - k_num) / k_num
        else:
            if k_num == 0.0:
                error = k_num - c_num
            else:
                error = (k_num - c_num) / k_num

        if maximum < error:
            maximum = error
            c_max = c_num
            k_max = k_num
    print("maximum error : " + str(maximum) + " when c has an element of " + str(c_max) +
          " and keras has an element of " + str(k_max))

    c_out.close()
    k_out.close()
