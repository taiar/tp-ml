<?php

  class run_analys {

    private $execution_size;
    private $epochs;
    private $classes;
    private $extra;

    private $file;
    private $file_handle;
    private $file_source;
    private $file_size;
    private $file_line;

    private $num_runs = 0;

    private $runs;

    public function __construct($file = '', $classes = 10, $epochs = 30, $extra = 8) {
      $this->classes = $classes;
      $this->epochs = $epochs;
      $this->extra = $extra;
      $this->execution_size = $this->extra + $this->classes + $this->epochs;
      if(!empty($file)) $this->set_file($file);
    }

    public function set_file($file) {
      $this->file = $file;
      $this->file_source = file($this->file);
      $this->file_size = count($this->file_source);
      $this->file_line = 0;
      $this->num_runs = $this->file_size / $this->execution_size;
      $this->runs = array();
    }

    public function read() {
      for ($i=0; $i < $this->num_runs; $i++) {
        $run = new single_run();
        $this->line_rev(5);
        for ($j=0; $j < $this->epochs; $j++) {
          $run->read_epoch($this->line_read());
          $this->line_next();
        }
        $this->line_rev(3);
        for ($j=0; $j < $this->classes; $j++) {
          $run->read_class($this->line_read());
          $this->line_next();
        }
        $this->runs[] = $run;
      }
    }

    private function line_read() {
      return $this->file_source[$this->file_line];
    }

    private function line_next() {
      $this->file_line += 1;
    }

    private function line_rev($n) {
      $this->file_line += $n;
    }
  }

  class single_run {

    private $execution_size;
    private $epochs;
    private $classes;
    private $extra;

    private $epoch = array();
    private $class = array();

    public function __construct($classes = 10, $epochs = 30, $extra = 8) {
      $this->classes = $classes;
      $this->epochs = $epochs;
      $this->extra = $extra;
      $this->execution_size = $this->extra + $this->classes + $this->epochs;

      $this->epoch = array();
      $this->class = array();
    }

    public function read_epoch($s) {
      $base = strpos($s, "had ") + 4;
      $base_2 = strpos($s, "errors") + 8;
      $this->epoch[] = array(
        'num' => substr($s, $base, strpos(substr($s, $base), " ")),
        'por' => substr($s, $base_2, 6)
      );
    }

    public function read_class($s) {
      $base = strpos($s, "/");
      echo substr($s, strpos($s, "(") + 1, 6) . PHP_EOL;
      $this->class[] = array(
        'num' => substr($s, 3, $base - 3),
        'tot' => substr($s, $base + 1, strpos(substr($s, $base), "(") - 1),
        'por' => substr($s, strpos($s, "(") + 1, 6)
      );
    }

  }

  $a = new run_analys('3_4_saida');
  $a->read();