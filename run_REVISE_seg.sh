echo "Start REVISE jobs."

parts=(part1 part2 part3)
iterations=(0)
patient_id="P2CRC"
spot_sizes=(1 2 3 4)

real_sc_refs=(1)


result_dir=results


use_raw_flag=1
for part in "${parts[@]}"; do
  for iteration in "${iterations[@]}"; do
    mkdir -p 0_records/run_REVISE_seg/${patient_id}_${part}_${iteration}
    echo "Start patient_id: ${patient_id}; part ${part}. Data iteration ${iteration} ....."
    for real_sc_ref in "${real_sc_refs[@]}"; do
        for spot_size in "${spot_sizes[@]}"; do
            echo "spot_size: ${spot_size}, use_raw_flag: ${use_raw_flag}, real_sc_ref: ${real_sc_ref}"
            nohup python -u run_REVISE_seg.py --spot_size=${spot_size} \
                            --use_raw_flag=${use_raw_flag} \
                            --real_sc_ref=${real_sc_ref} \
                            --part=${part} \
                            --iteration=${iteration} \
                            --patient_id=${patient_id} \
                            --result_dir=${result_dir} > 0_records/run_REVISE_seg/${patient_id}_${part}/${spot_size}_${use_raw_flag}_${real_sc_ref}.log 2>&1 &
        done
        wait
    done
    # wait
  done
  wait
done



echo "Finished REVISE jobs. Check the logs in the records directory."

