parts=(part1 part2 part3)
spot_sizes=(20 50 100 200)

patient_id="P2CRC"

batch_num=(0 1 2 3 4)

result_dir=results

for spot_size in "${spot_sizes[@]}"; do
  for part in "${parts[@]}"; do
      mkdir -p 0_records/run_REVISE_spot/${patient_id}_${part}
      echo "Start patient_id: ${patient_id}; part ${part}; spot_size ${spot_size}....."
      nohup python -u run_REVISE.py --spot_size=${spot_size} \
                              --batch_num=${batch_num} \
                              --part=${part} \
                              --result_dir=${result_dir} > 0_records/run_REVISE_spot/${patient_id}_${part}/${spot_size}_${batch_num}.log 2>&1 &
      # wait
    done
    wait
  done
done


echo "Finished REVISE jobs. Check the logs in the records directory."

