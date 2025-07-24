parts=(part1 part2 part3)
spot_sizes=(20 50 100 200)
patient_id="P2CRC"

real_sc_refs=(0 1)

result_dir=results

use_raw_flag=1
for part in "${parts[@]}"; do
  for spot_size in "${spot_sizes[@]}"; do
    for real_sc_ref in "${real_sc_refs[@]}"; do
      mkdir -p 0_records/run_REVISE_spot/${patient_id}_${part}
      echo "Start patient_id: ${patient_id}; part ${part}."
      nohup python -u run_REVISE.py --spot_size=${spot_size} \
                              --use_raw_flag=${use_raw_flag} \
                              --real_sc_ref=${real_sc_ref} \
                              --part=${part} \
                              --result_dir=${result_dir} > 0_records/run_REVISE_spot/${patient_id}_${part}/${spot_size}_${use_raw_flag}_${real_sc_ref}.log 2>&1 &
      # wait
    done
    wait
  done
done


use_raw_flag=0
real_sc_ref=0 # 必须为0
for part in "${parts[@]}"; do
  for spot_size in "${spot_sizes[@]}"; do
    mkdir -p 0_records/run_REVISE_spot/${patient_id}_${part}
    echo "spot_size: ${spot_size}, use_raw_flag: ${use_raw_flag}, real_sc_ref: ${real_sc_ref}"
    
    nohup python -u run_REVISE.py --spot_size=${spot_size} \
                            --use_raw_flag=${use_raw_flag} \
                            --real_sc_ref=${real_sc_ref} \
                            --part=${part} \
                            --result_dir=${result_dir} > 0_records/run_REVISE_spot/${patient_id}_${part}/${spot_size}_${use_raw_flag}_${real_sc_ref}.log 2>&1 &
done
wait

echo "Finished REVISE jobs. Check the logs in the records directory."

