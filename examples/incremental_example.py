import holoclean
import logging
import sys
from detect import NullDetector, ViolationDetector
from repair.featurize import *
sys.path.append('../')

dataset_name = 'hospital'
# batches = ['1-100', '101-200', '201-300', '301-400', '401-500',
#            '501-600', '601-700', '701-800', '801-900', '901-1000']
batches = ['1-100', '101-200']

# This line pauses the execution to drop the tables if needed.
drop = None
while drop != 'y' and drop != 'n':
    drop = input('Do you want to drop tables <dataset>_repaired, single_attr_stats, and pair_attr_stats? (y/n) ')

# We may run out of memory if HoloClean is not reinstantiated at each loading step.
for batch in batches:
    # Setup a HoloClean session.
    hc = holoclean.HoloClean(
        db_name='holo',
        domain_thresh_1=0,
        domain_thresh_2=0,
        weak_label_thresh=0.99,
        max_domain=10000,
        cor_strength=0.6,
        nb_cor_strength=0.8,
        epochs=2,
        weight_decay=0.01,
        learning_rate=0.001,
        threads=1,
        batch_size=1,
        verbose=False,
        timeout=3 * 60000,
        feature_norm=False,
        weight_norm=False,
        print_fw=False,
        incremental=True,
        incremental_entropy=False,
        repair_previous_errors=True
    ).session

    if batch == batches[0]:
        if drop == 'y':
            hc.ds.engine.drop_tables([dataset_name + '_repaired', 'single_attr_stats', 'pair_attr_stats'])

    # Load existing data and Denial Constraints.
    hc.load_data(dataset_name, '../testdata/' + dataset_name + '_' + batch + '.csv')
    hc.load_dcs('../testdata/' + dataset_name + '_constraints.txt')
    hc.ds.set_constraints(hc.get_dcs())

    # Detect erroneous cells using these two detectors.
    detectors = [NullDetector(), ViolationDetector()]
    hc.detect_errors(detectors)

    pause = input('Calculate the total number of errors. Ready? (y/n) ')
    """
    --Count the total number of errors before repairing (first batch: only table)
    DO $$
    DECLARE
      c RECORD;
      cnt BIGINT;
      qry TEXT;
    BEGIN
      FOR c IN (SELECT column_name
           FROM information_schema.columns
           WHERE table_schema = 'public'
           AND table_name   = 'hospital') LOOP
        qry := 'SELECT count(*) FROM  hospital as t1, hospital_clean as t2 WHERE t1._tid_ = t2._tid_ 
                 AND t2._attribute_ = ''' || c.column_name || ''' AND t1."' || c.column_name || '"::text != t2._value_';
        EXECUTE qry INTO cnt;
        RAISE NOTICE E'% = %', c.column_name, cnt;
      END LOOP;
    END;
    $$ LANGUAGE plpgsql;
    
    --Count the total number of errors before repairing (subsequent batches: table + table_repaired)
    DO $$
    DECLARE
      c RECORD;
      cnt BIGINT;
      qry TEXT;
    BEGIN
      FOR c IN (SELECT column_name
           FROM information_schema.columns
           WHERE table_schema = 'public'
           AND table_name   = 'hospital') LOOP
        qry := 'WITH all_rows AS (SELECT * FROM hospital UNION SELECT * FROM hospital_repaired)
              SELECT count(*) FROM  all_rows as t1, hospital_clean as t2 WHERE t1._tid_ = t2._tid_ 
                AND t2._attribute_ = ''' || c.column_name || ''' AND t1."' || c.column_name || '"::text != t2._value_';
      EXECUTE qry INTO cnt;
      RAISE NOTICE E'% = %', c.column_name, cnt;
      END LOOP;
    END;
    $$ LANGUAGE plpgsql;
    """

    # Repair errors based on the defined features.
    hc.setup_domain()
    featurizers = [
        InitAttrFeaturizer(),
        OccurAttrFeaturizer(),
        FreqFeaturizer(),
        ConstraintFeaturizer()
    ]
    hc.repair_errors(featurizers)

    # Evaluate the correctness of the results.
    hc.evaluate(fpath='../testdata/' + dataset_name + '_clean.csv',
                tid_col='tid',
                attr_col='attribute',
                val_col='correct_val')

    logging.info('Batch %s finished', batch)

    pause = input('Count the number of remaining errors and update the stats for the repaired table. Ready? (y/n) ')
    """
    --Count the total number of errors after repairing (table_repaired)
    DO $$
    DECLARE
      c RECORD;
      cnt BIGINT;
      qry TEXT;
    BEGIN
      FOR c IN (SELECT column_name
           FROM information_schema.columns
           WHERE table_schema = 'public'
           AND table_name   = 'hospital') LOOP
        qry := 'SELECT count(*) FROM  hospital_repaired as t1, hospital_clean as t2 WHERE t1._tid_ = t2._tid_ 
                 AND t2._attribute_ = ''' || c.column_name || ''' AND t1."' || c.column_name || '"::text != t2._value_';
        EXECUTE qry INTO cnt;
        RAISE NOTICE E'% = %', c.column_name, cnt;
      END LOOP;
    END;
    $$ LANGUAGE plpgsql;    
    
    --update single_attr_stats
    DO $$
    DECLARE
      c RECORD;
      qry TEXT;
    BEGIN
      TRUNCATE TABLE single_attr_stats;
      FOR c IN (SELECT column_name
           FROM information_schema.columns
           WHERE table_schema = 'public'
           AND table_name   = 'hospital') LOOP
        qry := 'INSERT INTO single_attr_stats(attr, val, freq) 
                SELECT ''' || c.column_name || ''' AS attr, "' || c.column_name || '"::text AS valor, count(*) AS freq
                FROM  hospital_repaired as t1 GROUP BY "' || c.column_name || '"';
          --RAISE NOTICE '%', qry;
        EXECUTE qry;
      END LOOP;
    END;
    $$ LANGUAGE plpgsql;
    
    --update pair_attr_stats
    DO $$
    DECLARE
      c1 RECORD; c2 RECORD; c3 RECORD;
      qry_schema TEXT; qry TEXT;
    BEGIN
      TRUNCATE TABLE pair_attr_stats;
      qry_schema := 'SELECT column_name FROM information_schema.columns WHERE table_schema = ''public'' AND table_name = ''hospital''';
      FOR c1 IN EXECUTE qry_schema LOOP
        FOR c2 IN EXECUTE qry_schema LOOP
          qry := 'INSERT INTO pair_attr_stats(attr1, attr2, val1, val2, freq) 
              SELECT ''' || c1.column_name || ''' AS attr1, ''' || c2.column_name || ''' AS attr2, "'
            || c1.column_name || '"::text AS val1, "' || c2.column_name || '"::text AS val2, count(*) AS freq
            FROM  hospital_repaired as t1 GROUP BY "' || c1.column_name || '", "' || c2.column_name || '"';
          --RAISE NOTICE '%', qry;
        EXECUTE qry;
      END LOOP;
      END LOOP;
    END;
    $$ LANGUAGE plpgsql;
    """