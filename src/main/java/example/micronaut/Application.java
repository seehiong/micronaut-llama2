package example.micronaut;

import io.micronaut.context.ApplicationContext;
import io.micronaut.context.annotation.Value;
import io.micronaut.runtime.Micronaut;
import jakarta.inject.Singleton;

@Singleton
public class Application {

    private final String parallelism;

    public Application(@Value("${java.util.concurrent.ForkJoinPool.common.parallelism:8}") String parallelism) {
        this.parallelism = parallelism;
    }

    public void run(String[] args) {
        // Programmatically set the parallelism property
        System.setProperty("java.util.concurrent.ForkJoinPool.common.parallelism", parallelism);
        System.out.println("ForkJoinPool parallelism set to: " + System.getProperty("java.util.concurrent.ForkJoinPool.common.parallelism"));
    }

    public static void main(String[] args) {
        ApplicationContext context = Micronaut.run(Application.class, args);
        Application app = context.getBean(Application.class);
        app.run(args);
    }
}